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
    # 로컬용 .env 처리
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

st.set_page_config(page_title="회계감리 분석 시스템", layout="wide")

# ==========================================
# 2. 데이터 로드 함수
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM cases"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df.columns = [c.replace(' ', '') for c in df.columns]
    if '결정연도' in df.columns:
        df['결정연도'] = df['결정연도'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df = df[df['결정연도'] != '']
        df = df.sort_values('결정연도')

    def map_group(x):
        if pd.isna(x): return "📝 기타"
        t = str(x).replace(" ", "")
        if any(k in t for k in ['매출','수익','채권']): return "💰 매출·채권"
        elif any(k in t for k in ['재고','매출원가','매입']): return "📦 재고·매입"
        elif any(k in t for k in ['금융','주식','파생','현금']): return "🏦 금융·현금"
        elif any(k in t for k in ['유형','무형','손상']): return "🏗️ 유·무형자산"
        elif any(k in t for k in ['자본','잉여금']): return "💎 자본"
        elif any(k in t for k in ['법인세']): return "⚖️ 법인세"
        else: return "📝 기타"

    df['표준그룹'] = df['관련계정과목'].apply(map_group)
    return df

def save_ai_log(prompt, response):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS ai_logs (timestamp TEXT, prompt TEXT, response TEXT)''')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO ai_logs VALUES (?, ?, ?)", (timestamp, str(prompt), str(response)))
        conn.commit()
        conn.close()
    except: pass

df_all = load_data()

# ==========================================
# 3. 메인 화면
# ==========================================
st.title("📊 회계감리 지적사례 AI 분석 시스템")

tab1, tab2 = st.tabs(["1️⃣ 종합 개요", "2️⃣ 심화 분석 & 조회"])

with tab1:
    total = len(df_all)
    top = df_all['표준그룹'].mode()[0] if not df_all.empty else "-"
    c1, c2, c3 = st.columns(3)
    c1.metric("총 분석 파일", f"{total}건")
    c2.metric("최다 적발", top)
    c3.metric("AI 엔진", "Gemini 2.0 Flash (최신)")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if not df_all.empty:
            cnt = df_all['표준그룹'].value_counts().reset_index()
            cnt.columns = ['그룹','건수']
            st.plotly_chart(px.pie(cnt, values='건수', names='그룹', hole=0.4), use_container_width=True)
    with c2:
        kwd = st.text_input("키워드 검색", key="k1")
        if kwd:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(kwd).any(), axis=1)
            st.dataframe(df_all[mask][['파일명','회사명','관련계정과목']], use_container_width=True)
        else:
            st.dataframe(df_all[['파일명','회사명','관련계정과목']], use_container_width=True)

with tab2:
    col_main, col_side = st.columns([7, 3])
    
    # [왼쪽] 리포트 생성 (안정적인 기본 모델 사용)
    with col_main:
        st.markdown("### 🤖 계정별 리포트")
        grps = sorted(df_all['표준그룹'].unique()) if not df_all.empty else []
        target = st.selectbox("영역 선택", grps)
        
        if not df_all.empty:
            sub = df_all[df_all['표준그룹'] == target]
            st.info(f"'{target}' 관련 {len(sub)}건 발견")
            if not sub.empty:
                trend = sub['결정연도'].value_counts().sort_index().reset_index()
                trend.columns = ['연도','건수']
                st.plotly_chart(px.line(trend, x='연도', y='건수', markers=True), use_container_width=True)
        
        if st.button("🚀 리포트 생성"):
            with st.spinner("분석 중..."):
                try:
                    # 리포트 작성용 모델 (안정적인 2.0 Flash 사용)
                    report_model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    cases = ""
                    for i, r in sub.sort_values('결정연도', ascending=False).head(15).iterrows():
                        cases += f"- [{r['결정연도']}] {r['회사명']}: {r['지적사항요약']}\n"
                    
                    prompt = f"회계사 관점에서 '{target}' 감리사례 분석.\n[사례]\n{cases}\n목차: 1.트렌드 2.주요수법 3.감사절차(5가지)"
                    res = report_model.generate_content(prompt).text
                    st.markdown(res)
                    save_ai_log(f"{target} 리포트", res)
                except Exception as e:
                    st.error(f"에러: {e}")

    # [오른쪽] 기준서 조회 봇 (구글 검색 + Gemini 2.0)
    with col_side:
        with st.container(border=True):
            st.markdown("### 📘 기준서/감사기준 조회")
            
            # 구글 검색 사용 여부 스위치 (기본값 켜기)
            use_google = st.toggle("Google 실시간 검색 사용", value=True)
            
            q = st.text_input("질문 입력", placeholder="예: 재고자산 평가 기준")
            
            if q:
                msg = "구글 검색 중..." if use_google else "AI 답변 중..."
                with st.spinner(msg):
                    try:
                        if use_google:
                            try:
                                # [핵심 변경] 정기님 계정에 있는 'gemini-2.0-flash' 사용
                                tools = [{"google_search": {}}]
                                chat_model = genai.GenerativeModel('gemini-2.0-flash', tools=tools)
                                g_prompt = "Google 검색 도구를 활용하여 최신 기준서를 확인 후 답변하세요."
                            except:
                                st.warning("⚠️ 검색 기능 불가. 기본 지식으로 답변합니다.")
                                chat_model = genai.GenerativeModel('gemini-2.0-flash')
                                g_prompt = "당신의 지식을 바탕으로 답변하세요."
                        else:
                            chat_model = genai.GenerativeModel('gemini-2.0-flash')
                            g_prompt = "당신의 지식을 바탕으로 답변하세요."

                        final_prompt = f"""
                        당신은 회계 기준 전문가입니다. {g_prompt}
                        질문: {q}
                        
                        [답변 형식]
                        1. **관련 기준서**: (정확한 명칭 및 문단 번호)
                        2. **핵심 규정**: (요약)
                        3. **실무 유의사항**: (감사 포인트)
                        """
                        
                        # 스트리밍 없이 한 번에 받기
                        res = chat_model.generate_content(final_prompt, stream=False)
                        
                        if res.text:
                            st.markdown(res.text)
                            save_ai_log(f"기준서 검색(G={use_google}): {q}", res.text)
                        else:
                            st.error("답변 생성 실패")
                            
                    except Exception as e:
                        st.error(f"오류: {e}")
                        st.caption("팁: 'Google 검색 사용'을 끄고 시도해보세요.")