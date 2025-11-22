import streamlit as st
import pandas as pd
import sqlite3
import os
import google.generativeai as genai
import plotly.express as px 
from collections import Counter
import datetime

# ==========================================
# 1. 설정 (API 키 자동 감지: Cloud vs Local)
# ==========================================
try:
    # [방법 1] Streamlit Cloud 서버의 비밀 금고(Secrets) 확인
    MY_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # [방법 2] 로컬 컴퓨터의 .env 파일 확인
    try:
        from dotenv import load_dotenv
        load_dotenv()
        MY_API_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        MY_API_KEY = None

# 키가 없으면 경고 띄우고 멈춤
if not MY_API_KEY:
    st.error("🚨 API 키를 찾을 수 없습니다!")
    st.info("Tip: 로컬에서는 .env 파일을, 배포 시에는 Streamlit Secrets를 설정해주세요.")
    st.stop()

# 설정 완료
DB_FILE = "audit_database.db"
genai.configure(api_key=MY_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="회계감리 분석 시스템", layout="wide")

# ==========================================
# 2. 기능 함수들 (데이터 로드 & 매핑 & 로그)
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM cases"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 컬럼 정리
    df.columns = [c.replace(' ', '') for c in df.columns]
    
    # 연도 정리
    if '결정연도' in df.columns:
        df['결정연도'] = df['결정연도'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df = df[df['결정연도'] != '']
        df = df.sort_values('결정연도')

    # 실무 그룹 매핑
    def map_group(account_str):
        if pd.isna(account_str): return "📝 기타/주석"
        target = str(account_str).replace(" ", "")
        
        if any(x in target for x in ['매출', '수익', '채권', '미수', '대손']): return "💰 매출·채권"
        elif any(x in target for x in ['재고', '매출원가', '매입', '채무']): return "📦 재고·매입"
        elif any(x in target for x in ['금융', '주식', '파생', '투자', '현금']): return "🏦 금융·현금"
        elif any(x in target for x in ['유형', '무형', '감가', '손상', '부동산']): return "🏗️ 유·무형자산"
        elif any(x in target for x in ['자본', '잉여금', '주식보상']): return "💎 자본"
        elif any(x in target for x in ['법인세', '이연']): return "⚖️ 법인세"
        else: return "📝 기타/주석"

    df['표준그룹'] = df['관련계정과목'].apply(map_group)
    return df

def save_ai_log(prompt, response):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS ai_logs
                     (timestamp TEXT, prompt TEXT, response TEXT)''')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO ai_logs VALUES (?, ?, ?)", (timestamp, str(prompt), str(response)))
        conn.commit()
        conn.close()
    except:
        pass # 로그 저장 실패해도 메인 기능은 돌아가게 함

df_all = load_data()

# ==========================================
# 3. 메인 화면 구성
# ==========================================
st.title("📊 회계감리 지적사례 AI 분석 시스템")

tab1, tab2 = st.tabs(["1️⃣ 종합 개요 (Overview)", "2️⃣ 계정별 심화 분석 (Deep Dive)"])

# --------------------------------------------------------------------
# [탭 1] 종합 개요
# --------------------------------------------------------------------
with tab1:
    total_files = len(df_all)
    top_group = df_all['표준그룹'].mode()[0] if not df_all.empty else "-"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 분석 파일", f"{total_files} 건")
    col2.metric("최다 적발 영역", top_group)
    col3.metric("서버 상태", "온라인 (Online)")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎨 실무 그룹별 비중")
        if not df_all.empty:
            group_counts = df_all['표준그룹'].value_counts().reset_index()
            group_counts.columns = ['그룹', '건수']
            fig_pie = px.pie(group_counts, values='건수', names='그룹', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("🔎 간편 검색")
        kwd = st.text_input("키워드 입력 (파일명, 계정 등)", key="kwd1")
        if kwd:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(kwd).any(), axis=1)
            st.dataframe(df_all[mask][['파일명', '회사명', '관련계정과목']], use_container_width=True)
        else:
            st.dataframe(df_all[['파일명', '회사명', '관련계정과목']], use_container_width=True)

# --------------------------------------------------------------------
# [탭 2] 심화 분석 (챗봇 탑재)
# --------------------------------------------------------------------
with tab2:
    col_main, col_side = st.columns([7, 3])
    
    # [왼쪽] 메인 리포트
    with col_main:
        st.markdown("### 🤖 계정 그룹별 상세 리포트")
        
        unique_groups = sorted(df_all['표준그룹'].unique()) if not df_all.empty else []
        target_group = st.selectbox("분석할 업무 영역(Cycle)", unique_groups)
        
        if not df_all.empty:
            group_df = df_all[df_all['표준그룹'] == target_group]
            st.info(f"**'{target_group}'** 영역 통계: 총 {len(group_df)}건 적발")
            
            if not group_df.empty:
                trend = group_df['결정연도'].value_counts().sort_index().reset_index()
                trend.columns = ['연도', '건수']
                st.plotly_chart(px.line(trend, x='연도', y='건수', markers=True), use_container_width=True)

        st.markdown("---")
        
        if st.button("🚀 AI 감사 리포트 생성", type="primary"):
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                try:
                    stats = f"총 {len(group_df)}건. 최빈 위반: {group_df['위반유형'].mode()[0] if '위반유형' in group_df else '미상'}"
                    cases = ""
                    for idx, row in group_df.sort_values('결정연도', ascending=False).head(20).iterrows():
                         cases += f"- [{row['결정연도']}] {row['회사명']} ({row.get('위반유형','-')}): {row['지적사항요약']}\n"

                    prompt = f"""
                    당신은 회계법인 파트너입니다. '{target_group}' 감리 사례를 분석하여 리포트를 작성하세요.
                    [데이터] {stats}
                    [사례] {cases[:15000]}
                    
                    보고서 목차:
                    1. **Risk Overview**: 주요 부정 패턴 요약
                    2. **심층 사례 분석**: 주요 위반 수법 상세 설명
                    3. **Action Plan**: 감사인이 수행해야 할 구체적 절차 5가지
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    save_ai_log(f"{target_group} 리포트", response.text)
                except Exception as e:
                    st.error(f"오류: {e}")

    # [오른쪽] 기준서 조회 봇
    with col_side:
        with st.container(border=True):
            st.markdown("### 📘 기준서/감사기준 조회")
            st.caption("K-IFRS 및 감사기준에 근거하여 답변합니다.")
            
            user_query = st.text_input("검색어/질문 입력", placeholder="예: 수익인식 5단계")
            
            if user_query:
                with st.spinner("검색 중..."):
                    try:
                        ref_prompt = f"""
                        당신은 K-IFRS와 회계감사기준(KGAAS) 전문가입니다.
                        질문: {user_query}
                        
                        [답변 형식]
                        1. **관련 기준서**: (정확한 기준서 번호와 명칭)
                        2. **핵심 요약**: 규정 내용 설명
                        3. **실무 적용**: 감사 시 유의점
                        """
                        chat_response = model.generate_content(ref_prompt).text
                        st.markdown(chat_response)
                        save_ai_log(f"기준서 검색: {user_query}", chat_response)
                    except Exception as e:
                        st.error("오류 발생")