import streamlit as st
import pandas as pd
import sqlite3
import os
import google.generativeai as genai
import plotly.express as px 
import datetime

# ==========================================
# 1. 기본 설정
# ==========================================
try:
    MY_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        MY_API_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        MY_API_KEY = ""

if not MY_API_KEY:
    st.error("API 키가 없습니다.")
    st.stop()

DB_FILE = "audit_database.db"
genai.configure(api_key=MY_API_KEY)
target_model = 'gemini-2.5-flash-lite' # 1,500회 무료 모델

model = genai.GenerativeModel(target_model)

st.set_page_config(page_title="회계감리 AI 분석 시스템", layout="wide", page_icon="📊")

# 소스 숨기기
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 2. 로깅 및 데이터 로드
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

def save_ai_log(prompt, response):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('''CREATE TABLE IF NOT EXISTS ai_logs (timestamp TEXT, prompt TEXT, response TEXT)''')
        conn.execute("INSERT INTO ai_logs VALUES (?, ?, ?)", 
                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(prompt), str(response)))
        conn.commit(); conn.close()
    except: pass

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
        df = df.sort_values('결정연도', ascending=False)

    def map_category(row):
        t = (str(row.get('관련계정과목','')) + str(row.get('위반유형',''))).replace(" ", "")
        if '매출' in t or '수익' in t: return "매출/수익인식"
        if '재고' in t or '자산' in t: return "자산/재고자산"
        if '파생' in t or '금융' in t: return "금융/투자자산"
        if '횡령' in t or '배임' in t: return "횡령/배임"
        if '주석' in t: return "주석미기재"
        return "기타 회계이슈"
    df['이슈분류'] = df.apply(map_category, axis=1)
    return df

log_visit()
df_all = load_data()

# ==========================================
# 3. 화면 구성
# ==========================================
with st.sidebar:
    st.markdown("## 👨‍💻 Developer")
    st.info("**서정기 (Jeremy)**\n\n중앙대학교 경영학부\n(KICPA)")
    st.metric("누적 방문자", f"{get_visit_count()} 명")
    st.caption("Last Updated: 2025.12")

st.title("📊 회계감리 지적사례 AI 분석 시스템")
tab1, tab2 = st.tabs(["1️⃣ 개별 사례 검색 (PDF 뷰어)", "2️⃣ 테마별 통합 분석 & 기준서 챗봇"])

# [Tab 1] 개별 검색 (수정 없음)
with tab1:
    col_list, col_view = st.columns([1, 1.2])
    with col_list:
        kwd = st.text_input("키워드 입력", placeholder="예: 재고, 삼성")
        if kwd:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(kwd).any(), axis=1)
            filtered = df_all[mask]
        else: filtered = df_all
        filtered['Display'] = filtered['결정연도'] + " | " + filtered['회사명'] + " - " + filtered['지적사항요약'].str[:20] + "..."
        sel_val = st.selectbox("사례 선택:", filtered['Display'].unique())
    
    with col_view:
        if sel_val:
            row = filtered[filtered['Display'] == sel_val].iloc[0]
            with st.container(border=True):
                st.markdown(f"### 📌 {row['회사명']} ({row['결정연도']})")
                st.write(f"**위반유형:** {row.get('위반유형','-')} | **계정:** {row.get('관련계정과목','-')}")
                st.info(f"**⚠️ 지적:** {row['지적사항요약']}")
                st.warning(f"**💡 유의:** {row['감사인유의사항']}")
            
            st.markdown("---")
            # PDF 다운로드만 깔끔하게 (오류 방지)
            file_name = row.get('파일명', '')
            pdf_path = os.path.join("pdfs", str(file_name))
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 PDF 원본 다운로드", f, file_name=file_name, mime="application/pdf", use_container_width=True)
            else:
                st.error("⚠️ 원본 PDF 파일이 없습니다.")

# [Tab 2] 통합 분석 & 챗봇 부분 (덮어쓰기)
with tab2:
    col_analysis, col_bot = st.columns([1.5, 1])

    # [왼쪽] 통합 리포트
    with col_analysis:
        st.subheader("🤖 테마별 AI 리포트")
        target_kwd = st.text_input("주제 입력", placeholder="예: 건설업, 바이오, 횡령")
        
        if target_kwd:
            # 키워드 포함 사례 추출
            mask = df_all.apply(lambda x: x.astype(str).str.contains(target_kwd).any(), axis=1)
            target_df = df_all[mask]
            
            if not target_df.empty:
                st.success(f"관련 사례 {len(target_df)}건 발견")
                
                # 버튼을 눌러야만 실행
                if st.button("🚀 AI 리포트 생성 (클릭)"):
                    with st.spinner("사례들을 분석하여 리포트와 출처를 정리 중입니다..."):
                        try:
                            # 1. 프롬프트에 넣을 사례 텍스트 생성 (파일명 포함!)
                            # 최대 15개까지만 참조 (토큰 절약 및 속도)
                            cases_summary = ""
                            ref_list = [] # UI에 보여줄 참조 목록
                            
                            for i, r in target_df.head(15).iterrows():
                                # [파일명] 회사명: 내용 형식으로 구성
                                file_ref = r.get('파일명', '파일명미상')
                                case_text = f"- [출처: {file_ref}] {r['회사명']} ({r['결정연도']}): {r['지적사항요약']}"
                                cases_summary += case_text + "\n"
                                ref_list.append(f"{r['회사명']} ({file_ref})")
                            
                            # 2. 프롬프트 작성 (출처 표기 지시 강화)
                            prompt = f"""
                            당신은 회계법인 품질관리실 파트너입니다.
                            주제: **'{target_kwd}'** 관련 감리지적사례 종합 분석 리포트 작성.
                            
                            [분석 대상 데이터 (Source Data)]
                            {cases_summary}

                            [작성 요구사항]
                            1. **Risk Overview**: 해당 이슈가 회계감사에서 왜 위험한지 요약.
                            2. **Common Fraud Schemes**: 주요 회계부정/오류 수법 분석.
                            3. **Key Audit Procedures**: 감사인이 반드시 수행해야 할 절차 5가지.
                            4. **Reference**: 분석 내용 중간중간에 **(출처: FSS...pdf)** 형식으로 근거를 인용할 것.
                            
                            * 톤앤매너: 전문가답게 논리적으로 작성.
                            """
                            
                            # 3. AI 생성
                            response = model.generate_content(prompt).text
                            
                            # 4. 결과 출력
                            st.markdown(response)
                            
                            # 5. [New] 하단에 '참고한 파일 목록' 별도 표시 (신뢰도 UP)
                            with st.expander("📚 이 리포트가 참고한 원본 파일 목록 보기"):
                                for ref in ref_list:
                                    st.caption(f"• {ref}")
                            
                            # 로그 저장
                            save_ai_log(f"리포트(RAG): {target_kwd}", response)
                            
                        except Exception as e: st.error(f"오류: {e}")
            else:
                st.warning("해당 키워드로 검색된 사례가 없습니다.")

    # [오른쪽] 기준서 챗봇 (기존 코드 유지)
    with col_bot:
        st.markdown("### 📘 기준서 챗봇")

        with st.form(key='chat_form'):
            user_q = st.text_input("질문 입력", placeholder="예: 재고자산 실사")
            submit_button = st.form_submit_button(label='질문하기')

        if submit_button and user_q:
            with st.spinner("답변 생성 중..."):
                try:
                    prompt = f"질문: {user_q}\n근거가 되는 기준서 문단 번호를 꼭 포함해서 설명해줘."
                    res = model.generate_content(prompt).text

                    st.markdown(res)
                    save_ai_log(f"챗봇: {user_q}", res)
                except Exception as e: st.error(f"오류: {e}")
