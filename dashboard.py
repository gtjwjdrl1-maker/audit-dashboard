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

target_model = 'gemini-2.0-flash'
try:
    tools = [{"google_search": {}}]
    model = genai.GenerativeModel(target_model, tools=tools)
except:
    model = genai.GenerativeModel(target_model)

st.set_page_config(page_title="회계감리 분석 시스템", layout="wide")

# ==========================================
# 2. 데이터 로드 및 [고도화된 분류 매핑]
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

    # [핵심 수정] 더 촘촘해진 상세 분류 로직
    def map_detailed_group(row):
        # 검색 대상 텍스트 확장 (관련계정 + 위반유형 + 요약)
        t = (str(row.get('관련계정과목','')) + str(row.get('위반유형','')) + str(row.get('지적사항요약',''))).replace(" ", "")
        
        # 1. 최우선 적발 (부정/오류)
        if any(x in t for x in ['횡령', '배임', '가공자산', '유용']): return "🚨 횡령·배임 및 자금유용"
        if any(x in t for x in ['분식', '조작', '허위', '가공매출']): return "💣 고의적 회계분식/조작"
        
        # 2. 매출/채권 (Revenue Cycle)
        if any(x in t for x in ['매출채권', '대손', '채권', '충당금', '회수']): return "💰 매출채권/대손충당금 (AR)"
        if any(x in t for x in ['매출', '수익', '공사', '진행률', '인도', '총액', '순액']): return "📊 매출/수익인식 (Revenue)"
        
        # 3. 자산 (Asset)
        if any(x in t for x in ['개발비', '무형', '영업권', '손상']): return "💡 무형자산/개발비 과대계상"
        if any(x in t for x in ['재고', '평가손실', '저가법', '진부화', '수불']): return "📦 재고자산 평가/실재성"
        if any(x in t for x in ['유형', '감가', '토지', '건물', '기계', '리스', '사용권']): return "🏗️ 유형자산/감가상각"
        
        # 4. 금융/투자 (Finance)
        if any(x in t for x in ['파생', '전환사채', 'RCPS', '금융상품', '옵션', 'BW', 'CB']): return "📉 파생상품/복합금융상품"
        if any(x in t for x in ['종속', '관계', '지분법', '주식', '투자주식', '펀드']): return "📈 투자주식/지분법 평가"
        if any(x in t for x in ['대여금', '선급금', '가지급금', '현금', '예금']): return "💸 대여금/자금거래"
        
        # 5. 부채/자본 (Liabilities/Equity)
        if any(x in t for x in ['차입금', '매입채무', '미지급', '부채', '충당부채', '보증']): return "📉 차입금/우발부채"
        if any(x in t for x in ['자본', '잉여금', '주식보상', '스톡옵션', '신주', '자기주식']): return "💎 자본/주식보상비용"
        if any(x in t for x in ['합병', '사업결합', '인수']): return "🤝 합병/사업결합 (M&A)"
        
        # 6. 세무/공시 (Tax/Disclosure)
        if any(x in t for x in ['법인세', '이연']): return "⚖️ 법인세회계"
        if any(x in t for x in ['주석', '담보', '약정']): return "📝 주석 미기재 (공시)"
        if any(x in t for x in ['특수관계', '이해관계']): return "🔗 특수관계자 거래"
        
        return "🔍 기타 일반 회계처리"

    def map_group(x): # 대분류용 (1페이지 차트용)
        d = map_detailed_group({'관련계정과목':x, '위반유형':x, '지적사항요약':x}) # 약식 매핑
        if '매출' in d or '수익' in d: return "💰 매출·채권"
        if '재고' in d or '자산' in d: return "🏗️ 자산·재고"
        if '금융' in d or '투자' in d or '파생' in d: return "🏦 금융·투자"
        if '부채' in d or '자본' in d: return "⚖️ 부채·자본"
        if '횡령' in d or '분식' in d: return "🚨 부정·오류"
        else: return "📝 공시·기타"

    df['상세분류'] = df.apply(map_detailed_group, axis=1)
    df['표준그룹'] = df['관련계정과목'].apply(map_group)
    return df

def save_ai_log(prompt, response):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO ai_logs VALUES (?, ?, ?)", 
                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(prompt), str(response)))
        conn.commit(); conn.close()
    except: pass

# ==========================================
# 3. 방문자 집계 및 데이터 로드
# ==========================================
def log_visit():
    if 'visited' not in st.session_state:
        try:
            conn = sqlite3.connect(DB_FILE)
            conn.execute('''CREATE TABLE IF NOT EXISTS visit_logs (timestamp TEXT)''')
            conn.execute("INSERT INTO visit_logs VALUES (?)", (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
            conn.commit(); conn.close()
            st.session_state['visited'] = True
        except: pass

def get_visit_count():
    try:
        conn = sqlite3.connect(DB_FILE)
        cnt = conn.execute("SELECT COUNT(*) FROM visit_logs").fetchone()[0]
        conn.close()
        return cnt
    except: return 0

def log_action(action_type, details):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('''CREATE TABLE IF NOT EXISTS user_actions (timestamp TEXT, action_type TEXT, details TEXT)''')
        conn.execute("INSERT INTO user_actions VALUES (?, ?, ?)", 
                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action_type, details))
        conn.commit(); conn.close()
    except: pass

def get_top_rankings():
    try:
        conn = sqlite3.connect(DB_FILE)
        df_c = pd.read_sql("SELECT details as '사례명', COUNT(*) as '조회수' FROM user_actions WHERE action_type='VIEW_CASE' GROUP BY details ORDER BY 조회수 DESC LIMIT 5", conn)
        df_k = pd.read_sql("SELECT prompt as '키워드', COUNT(*) as '질문수' FROM ai_logs GROUP BY prompt ORDER BY 질문수 DESC LIMIT 5", conn)
        conn.close()
        return df_c, df_k
    except: return pd.DataFrame(), pd.DataFrame()

log_visit()
df_all = load_data()

# ==========================================
# 4. 화면 구성
# ==========================================
with st.sidebar:
    st.markdown("## 👨‍💻 Developer")
    st.info("**서정기 (Jeremy)**\n\n중앙대학교 경영학부\n(KICPA)")
    st.metric("누적 방문자", f"{get_visit_count()} 명")
    st.caption("© 2025 All rights reserved.")

st.title("📊 회계감리 지적사례 AI 분석 시스템")

tab1, tab2 = st.tabs(["1️⃣ 종합 개요 (Trending)", "2️⃣ 심화 분석 (Deep Dive)"])

# [탭 1]
with tab1:
    total = len(df_all)
    top = df_all['상세분류'].mode()[0] if not df_all.empty else "-"
    top_cases, top_keywords = get_top_rankings()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 분석 파일", f"{total}건")
    col2.metric("최다 빈출 이슈", top) # 상세분류로 변경하여 더 구체적으로 보여줌
    hot_kwd = top_keywords.iloc[0]['키워드'] if not top_keywords.empty else "-"
    col3.metric("🔥 실시간 인기 질문", hot_kwd)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔥 많이 본 사례 Top 5")
        if not top_cases.empty:
            st.plotly_chart(px.bar(top_cases, x='조회수', y='사례명', orientation='h', text='조회수'), use_container_width=True)
        else: st.info("데이터 집계 중...")
    with c2:
        st.subheader("🤖 자주 묻는 질문 Top 5")
        if not top_keywords.empty:
            st.plotly_chart(px.bar(top_keywords, x='질문수', y='키워드', orientation='h', text='질문수', color='질문수'), use_container_width=True)
        else: st.info("데이터 집계 중...")

    st.markdown("---")
    st.subheader("🔎 전체 사례 검색")
    cl, cd = st.columns([1, 1])
    with cl:
        kwd = st.text_input("키워드 검색", key="search")
        if kwd:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(kwd).any(), axis=1)
            filtered = df_all[mask]
        else: filtered = df_all
        st.caption(f"결과: {len(filtered)}건")
        filtered['Label'] = filtered['회사명'] + " (" + filtered['결정연도'] + ") - " + filtered['상세분류']
        sel = st.selectbox("사례 선택", filtered['Label'].unique())
        if sel:
            if 'last_viewed' not in st.session_state or st.session_state['last_viewed'] != sel:
                log_action("VIEW_CASE", sel)
                st.session_state['last_viewed'] = sel
    with cd:
        if sel:
            row = filtered[filtered['Label'] == sel].iloc[0]
            st.info(f"📌 {row['회사명']} ({row['결정연도']})")
            st.write(f"**이슈:** {row['상세분류']} | **계정:** {row['관련계정과목']}")
            with st.container(border=True): st.write("**⚠️ 지적:** " + row['지적사항요약'])
            with st.container(border=True): st.success("**💡 유의:** " + row['감사인유의사항'])
            with st.expander("원문 보기"): st.text(row.get('원본텍스트(일부)', '내용 없음'))

# [탭 2]
with tab2:
    cm, cs = st.columns([7, 3])
    with cm:
        st.markdown("### 🤖 위반 유형별 심층 리포트")
        cats = sorted(df_all['상세분류'].unique())
        target = st.selectbox("분석할 핵심 이슈(Issue) 선택", cats)
        sub = df_all[df_all['상세분류'] == target]
        
        st.success(f"👉 **'{target}'** 관련 사례: {len(sub)}건")
        
        if not sub.empty:
            c1, c2 = st.columns(2)
            with c1:
                trend = sub['결정연도'].value_counts().sort_index().reset_index()
                trend.columns = ['연도','건수']
                st.plotly_chart(px.line(trend, x='연도', y='건수', title="연도별 추이"), use_container_width=True)
            with c2:
                if '위반유형' in sub.columns:
                    t_cnt = sub['위반유형'].value_counts().head(5).reset_index()
                    t_cnt.columns = ['유형','건수']
                    st.plotly_chart(px.pie(t_cnt, values='건수', names='유형', hole=0.4, title="주요 위반유형"), use_container_width=True)

        if st.button("🚀 리포트 생성"):
            with st.spinner("분석 중..."):
                try:
                    cases_txt = ""
                    for i, r in sub.sort_values('결정연도', ascending=False).head(20).iterrows():
                        cases_txt += f"- [{r['결정연도']}] {r['회사명']}: {r['지적사항요약']}\n"
                    
                    prompt = f"""
                    당신은 회계법인 파트너입니다. '{target}' 이슈를 분석하세요.
                    [사례] {cases_txt[:15000]}
                    [목차] 1.발생원인 2.주요수법 3.체크리스트(5개)
                    """
                    res = genai.GenerativeModel(target_model).generate_content(prompt).text
                    st.markdown(res)
                    save_ai_log(f"{target} 리포트", res)
                except Exception as e: st.error(f"오류: {e}")

    with cs:
        st.markdown("### 📘 기준서/감사기준 조회")
        std_type = st.radio("검색 대상", ["전체", "K-IFRS", "KGAAS"])
        use_g = st.toggle("Google 검색", value=True)
        q = st.text_input("질문 입력")
        
        if q:
            with st.spinner(f"{std_type} 검색 중..."):
                try:
                    if std_type == "K-IFRS":
                        role = "K-IFRS 전문가"
                        prefix = "K-IFRS"
                    elif std_type == "KGAAS":
                        role = "회계감사기준 전문가"
                        prefix = "회계감사기준"
                    else:
                        role = "회계 및 감사 전문가"
                        prefix = "K-IFRS 및 감사기준"

                    strict_p = f"""
                    당신은 {role}입니다. 질문: {q}
                    [지침] 기준서/문단 번호 필수 명시. 원문 인용.
                    """
                    
                    if use_g:
                        tools = [{"google_search": {}}]
                        m = genai.GenerativeModel(target_model, tools=tools)
                        final_p = f"Google 검색: '{prefix} {q} 문단'\n{strict_p}"
                    else:
                        m = genai.GenerativeModel(target_model)
                        final_p = strict_p
                    
                    r = m.generate_content(final_p, stream=False).text
                    st.markdown(r)
                    save_ai_log(f"챗봇({std_type}): {q}", r)
                except Exception as e: st.error(f"오류: {e}")