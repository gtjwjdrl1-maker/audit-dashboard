import streamlit as st
import pandas as pd
import sqlite3
import os
import google.generativeai as genai
import plotly.express as px 
from collections import Counter
import datetime

# ==========================================
# 1. 설정
# ==========================================
try:
    MY_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        MY_API_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        MY_API_KEY = "여기에_아까_복사한_키를_붙여넣으세요"

if not MY_API_KEY:
    st.error("API 키가 없습니다.")
    st.stop()

DB_FILE = "audit_database.db"
genai.configure(api_key=MY_API_KEY)

# 정기님 계정 전용 모델 (2.0 Flash)
target_model = 'gemini-2.0-flash'

try:
    tools = [{"google_search": {}}]
    model = genai.GenerativeModel(target_model, tools=tools)
except:
    model = genai.GenerativeModel(target_model)

st.set_page_config(page_title="회계감리 분석 시스템", layout="wide")

# ==========================================
# 2. 데이터 로드 및 [정밀 분류 매핑]
# ==========================================
@st.cache_data(ttl=0) 
def load_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM cases", conn)
    conn.close()
    
    df.columns = [c.replace(' ', '') for c in df.columns]
    if '결정연도' in df.columns:
        df['결정연도'] = df['결정연도'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df = df[df['결정연도'] != '']
        df = df.sort_values('결정연도')

    # [핵심 수정] 요청하신 5대 사이클 & 키워드 매핑 로직 적용
    def map_detailed_group(row):
        # 검색 대상 텍스트 (계정 + 위반유형 + 요약)
        t = (str(row.get('관련계정과목','')) + str(row.get('위반유형','')) + str(row.get('지적사항요약',''))).replace(" ", "")
        
        # 1. 매출/채권 Cycle
        if any(x in t for x in ['매출채권', '대손', '채권', '충당금']): return "💰 매출채권/대손 (AR)"
        if any(x in t for x in ['매출', '수익', '공사수익', '진행률']): return "📊 매출 허위/과대계상 (Revenue)"
        
        # 2. 자산/비용 Cycle
        if any(x in t for x in ['개발비', '무형', '영업권']): return "💡 무형자산/개발비 (Intangible)"
        if any(x in t for x in ['재고', '평가손실', '저가법']): return "📦 재고자산 이슈 (Inventory)"
        if any(x in t for x in ['유형', '감가', '토지', '건물']): return "🏗️ 유형자산/감가상각 (Tangible)"
        
        # 3. 금융/투자 Cycle
        if any(x in t for x in ['종속', '관계', '지분법', '주식']): return "📈 투자주식 평가 (Investment)"
        if any(x in t for x in ['파생', '전환사채', 'RCPS', '금융상품']): return "📉 파생/금융상품 (Derivatives)"
        if any(x in t for x in ['대여금', '선급금', '가지급금']): return "💸 대여금/선급금 (Loans)"
        
        # 4. 공시/주석 Cycle
        if any(x in t for x in ['주석', '담보', '약정', '우발']): return "📝 주석 미기재 (Disclosure)"
        if any(x in t for x in ['특수관계', '이해관계']): return "🤝 특수관계자 거래 (Related Party)"
        
        # 5. 기타 부정 Cycle
        if any(x in t for x in ['횡령', '배임', '가공자산']): return "🚨 횡령/배임 은폐 (Fraud)"
        if any(x in t for x in ['연결', '종속회사']): return "🔗 연결 범위 오류 (Consolidation)"
        
        return "🔍 기타 회계처리 (Others)"

    df['상세분류'] = df.apply(map_detailed_group, axis=1)
    return df

def save_ai_log(prompt, response):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO ai_logs VALUES (?, ?, ?)", 
                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(prompt), str(response)))
        conn.commit(); conn.close()
    except: pass

df_all = load_data()

# ==========================================
# 3. 메인 화면
# ==========================================
st.title("📊 회계감리 지적사례 AI 분석 시스템")

tab1, tab2 = st.tabs(["1️⃣ 사례 검색", "2️⃣ 계정별 심화 리포트"])

with tab1:
    col_list, col_detail = st.columns([1, 1])
    with col_list:
        kwd = st.text_input("키워드 검색", key="search")
        if kwd:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(kwd).any(), axis=1)
            filtered = df_all[mask]
        else: filtered = df_all
        
        st.caption(f"결과: {len(filtered)}건")
        # 라벨에 상세분류 표시
        filtered['Label'] = filtered['회사명'] + " (" + filtered['결정연도'] + ") - " + filtered['상세분류']
        sel = st.selectbox("사례 선택", filtered['Label'].unique())

    with col_detail:
        if sel:
            row = filtered[filtered['Label'] == sel].iloc[0]
            st.info(f"📌 {row['회사명']} ({row['결정연도']})")
            st.write(f"**이슈 분류:** {row['상세분류']}")
            st.write(f"**관련 계정:** {row['관련계정과목']}")
            with st.container(border=True): st.write("**⚠️ 지적:** " + row['지적사항요약'])
            with st.container(border=True): st.success("**💡 유의:** " + row['감사인유의사항'])
            with st.expander("원문 보기"): st.text(row.get('원본텍스트(일부)', '내용 없음'))

with tab2:
    col_m, col_s = st.columns([7, 3])
    
    # [왼쪽] 심화 분석
    with col_m:
        st.markdown("### 🤖 이슈별 심층 리포트")
        # 상세분류 기준으로 선택박스 구성
        cats = sorted(df_all['상세분류'].unique())
        target = st.selectbox("분석할 핵심 이슈(Issue) 선택", cats)
        sub = df_all[df_all['상세분류'] == target]
        
        st.success(f"👉 **'{target}'** 관련 사례: {len(sub)}건")
        
        if not sub.empty:
            c1, c2 = st.columns(2)
            with c1:
                trend = sub['결정연도'].value_counts().sort_index().reset_index()
                trend.columns = ['연도','건수']
                st.plotly_chart(px.line(trend, x='연도', y='건수'), use_container_width=True)
            with c2:
                if '위반유형' in sub.columns:
                    t_cnt = sub['위반유형'].value_counts().head(5).reset_index()
                    t_cnt.columns = ['유형','건수']
                    st.plotly_chart(px.pie(t_cnt, values='건수', names='유형', hole=0.4), use_container_width=True)

        if st.button("🚀 리포트 생성"):
            with st.spinner("분석 중..."):
                try:
                    cases_txt = ""
                    for i, r in sub.sort_values('결정연도', ascending=False).head(20).iterrows():
                        cases_txt += f"- [{r['결정연도']}] {r['회사명']}: {r['지적사항요약']}\n"
                    
                    prompt = f"""
                    당신은 회계법인 파트너입니다. '{target}' 이슈를 분석하세요.
                    [사례] {cases_txt[:15000]}
                    [목차] 1.발생원인 2.주요수법 3.감사체크리스트(5개)
                    """
                    res = genai.GenerativeModel(target_model).generate_content(prompt).text
                    st.markdown(res)
                    save_ai_log(f"{target} 리포트", res)
                except Exception as e: st.error(f"오류: {e}")

    # [오른쪽] 기준서 봇 (기준 선택 기능 추가!)
    with col_s:
        st.markdown("### 📘 기준서/감사기준 조회")
        
        # [핵심 수정] 검색 대상 선택 버튼 추가
        std_type = st.radio("검색 대상 기준 선택", ["전체 통합", "회계기준 (K-IFRS)", "감사기준 (KGAAS)"])
        
        use_g = st.toggle("Google 검색 사용", value=True)
        q = st.text_input("질문 입력", placeholder="예: 재고자산 실사 절차")
        
        if q:
            with st.spinner(f"{std_type} 검색 중..."):
                try:
                    # 선택된 기준에 따라 페르소나와 검색어 변경
                    if std_type == "회계기준 (K-IFRS)":
                        role = "K-IFRS(한국채택국제회계기준) 전문가"
                        search_prefix = "K-IFRS"
                    elif std_type == "감사기준 (KGAAS)":
                        role = "회계감사기준(KGAAS) 전문가"
                        search_prefix = "회계감사기준"
                    else:
                        role = "회계 및 감사 기준 통합 전문가"
                        search_prefix = "K-IFRS 및 회계감사기준"

                    strict_prompt = f"""
                    당신은 {role}입니다. 질문에 대해 관련 기준서 원문을 근거로 답변하세요.
                    
                    질문: {q}
                    
                    [필수 지침]
                    1. 반드시 **기준서 번호(제xxxx호)**와 **문단 번호**를 명시하세요.
                    2. 블로그 글이 아닌, 법령/기준서 원문을 인용하세요.
                    """
                    
                    if use_g:
                        tools = [{"google_search": {}}]
                        m = genai.GenerativeModel(target_model, tools=tools)
                        # 검색어에 'K-IFRS' 또는 '감사기준'을 강제로 붙여서 검색 정확도 향상
                        final_p = f"Google 검색 키워드: '{search_prefix} {q} 문단'\n{strict_prompt}"
                    else:
                        m = genai.GenerativeModel(target_model)
                        final_p = strict_prompt
                    
                    r = m.generate_content(final_p, stream=False).text
                    st.markdown(r)
                    save_ai_log(f"챗봇({std_type}): {q}", r)
                    
                except Exception as e: st.error(f"오류: {e}")