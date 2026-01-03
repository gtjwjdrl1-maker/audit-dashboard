import streamlit as st
import pandas as pd
import sqlite3
import os
import google.generativeai as genai
import plotly.express as px 
import datetime
import base64

# ==========================================
# 1. 기본 설정 및 보안
# ==========================================
try:
    MY_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        MY_API_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        MY_API_KEY = "" # 로컬 테스트 시 직접 입력

if not MY_API_KEY:
    st.error("API 키가 설정되지 않았습니다.")
    st.stop()

# DB 파일 경로
DB_FILE = "audit_database.db"

# Gemini 설정
genai.configure(api_key=MY_API_KEY)
target_model = 'gemini-2.0-flash' 
try:
    tools = [{"google_search": {}}]
    model = genai.GenerativeModel(target_model, tools=tools)
except:
    model = genai.GenerativeModel(target_model)

st.set_page_config(page_title="회계감리 AI 분석 시스템", layout="wide", page_icon="📊")

# 소스코드 숨기기 (보안)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 2. 데이터 로드 및 로깅 함수
# ==========================================
# 방문자 집계
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

# AI 질문 로그 저장
def save_ai_log(prompt, response):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('''CREATE TABLE IF NOT EXISTS ai_logs (timestamp TEXT, prompt TEXT, response TEXT)''')
        conn.execute("INSERT INTO ai_logs VALUES (?, ?, ?)", 
                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(prompt), str(response)))
        conn.commit(); conn.close()
    except: pass

# 데이터 로드 (캐싱)
@st.cache_data(ttl=0) 
def load_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM cases", conn)
    conn.close()
    
    # 컬럼 공백 제거
    df.columns = [c.replace(' ', '') for c in df.columns]
    
    # 연도 데이터 정제
    if '결정연도' in df.columns:
        df['결정연도'] = df['결정연도'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df = df[df['결정연도'] != '']
        df = df.sort_values('결정연도', ascending=False) # 최신순 정렬

    # 상세 분류 매핑 (차트용)
    def map_category(row):
        t = (str(row.get('관련계정과목','')) + str(row.get('위반유형',''))).replace(" ", "")
        if '매출' in t or '수익' in t: return "매출/수익인식"
        if '재고' in t or '자산' in t: return "자산/재고자산"
        if '파생' in t or '금융' in t or '주식' in t: return "금융/투자자산"
        if '횡령' in t or '배임' in t: return "횡령/배임"
        if '주석' in t: return "주석미기재"
        return "기타 회계이슈"

    df['이슈분류'] = df.apply(map_category, axis=1)
    return df

log_visit()
df_all = load_data()

# ==========================================
# 3. 사이드바 (개발자 정보)
# ==========================================
with st.sidebar:
    st.markdown("## 👨‍💻 Developer")
    st.info("**서정기 (Jeremy)**\n\n중앙대학교 경영학부\n(KICPA)")
    st.metric("누적 방문자", f"{get_visit_count()} 명")
    st.caption("Last Updated: 2025.12")
    st.markdown("---")
    st.markdown("### 📌 사용 가이드")
    st.markdown("""
    **Tab 1:** 개별 감리지적사례 검색 및 **PDF 원본 열람**
    **Tab 2:** 키워드 기반 **AI 통합 리포트** 작성 & 기준서 챗봇
    """)

st.title("📊 회계감리 지적사례 AI 분석 시스템")

# 탭 구성
tab1, tab2 = st.tabs(["1️⃣ 개별 사례 검색 (PDF 뷰어)", "2️⃣ 테마별 통합 분석 & 기준서 챗봇"])

# ==============================================================================
# [TAB 1] 개별 사례 검색 및 PDF 뷰어
# ==============================================================================
with tab1:
    col_list, col_view = st.columns([1, 1.2]) # 화면 분할 (왼쪽:검색 / 오른쪽:뷰어)

    # [왼쪽] 검색 및 목록
    with col_list:
        st.subheader("🔎 사례 검색")
        kwd = st.text_input("키워드 입력 (예: 재고, 삼성, 횡령)", placeholder="검색어 입력...")
        
        # 필터링 로직
        if kwd:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(kwd).any(), axis=1)
            filtered = df_all[mask]
        else:
            filtered = df_all

        st.caption(f"검색 결과: {len(filtered)}건")
        
        # 선택 박스 (최신순)
        filtered['Display'] = filtered['결정연도'] + " | " + filtered['회사명'] + " - " + filtered['지적사항요약'].str[:20] + "..."
        sel_val = st.selectbox("열람할 사례를 선택하세요:", filtered['Display'].unique())
    
    # [오른쪽] 상세 정보 및 PDF 뷰어
    with col_view:
        if sel_val:
            # 선택된 행 데이터 가져오기
            row = filtered[filtered['Display'] == sel_val].iloc[0]
            
            # 1. 핵심 요약 카드
            with st.container(border=True):
                st.markdown(f"### 📌 {row['회사명']} ({row['결정연도']})")
                st.write(f"**위반유형:** {row.get('위반유형','-')} | **관련계정:** {row.get('관련계정과목','-')}")
                st.info(f"**⚠️ 지적사항:** {row['지적사항요약']}")
                st.warning(f"**💡 감사인 유의사항:** {row['감사인유의사항']}")

            # 2. PDF 원본 뷰어 (수정된 코드)
            st.markdown("---")
            st.subheader("📄 감리지적사례 원본(PDF)")
            
            # 파일 경로
            file_name = row.get('파일명', '')
            pdf_path = os.path.join("pdfs", str(file_name))
            
            if os.path.exists(pdf_path) and str(file_name).lower().endswith('.pdf'):
                # (1) PDF 파일 읽기
                with open(pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                
                # (2) [수정] 다운로드 버튼을 먼저, 더 크게 보여줌 (가장 확실한 방법)
                st.download_button(
                    label="📥 PDF 원본 파일 다운로드 (미리보기가 안 보이면 클릭)",
                    data=open(pdf_path, "rb"),
                    file_name=file_name,
                    mime="application/pdf",
                    use_container_width=True  # 버튼을 꽉 차게 만들어서 강조
                )

                # (3) [수정] iframe 대신 embed 태그 사용 (호환성 개선)
                # 일부 브라우저 차단 메시지 방지를 위한 안내 문구 추가
                st.caption("※ 브라우저 보안 설정에 따라 미리보기가 차단될 수 있습니다. 위 다운로드 버튼을 이용해주세요.")
                
                pdf_display = f'''
                    <embed 
                        src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" 
                        height="800" 
                        type="application/pdf"
                    >
                '''
                st.markdown(pdf_display, unsafe_allow_html=True)
                
            else:
                st.error("⚠️ 원본 PDF 파일을 찾을 수 없습니다.")

# ==============================================================================
# [TAB 2] 키워드 기반 통합 분석 & 기준서 챗봇
# ==============================================================================
with tab2:
    col_analysis, col_bot = st.columns([1.5, 1])

    # [왼쪽] 키워드 통합 분석 리포트
    with col_analysis:
        st.subheader("🤖 키워드 기반 AI 심층 리포트")
        st.markdown("특정 **산업(건설, 제약)**이나 **이슈(무형자산, 특수관계자)**를 입력하면, 관련 사례를 모두 모아 분석합니다.")
        
        target_kwd = st.text_input("분석 주제 입력", placeholder="예: 건설업, 바이오, 지주사, 파생상품...")
        
        if target_kwd:
            # 키워드 포함 사례 추출
            mask = df_all.apply(lambda x: x.astype(str).str.contains(target_kwd).any(), axis=1)
            target_df = df_all[mask]
            
            if not target_df.empty:
                st.success(f"👉 **'{target_kwd}'** 관련 사례 총 **{len(target_df)}건**을 발견했습니다.")
                
                # 시각화 (연도별 추이)
                trend = target_df['결정연도'].value_counts().sort_index().reset_index()
                trend.columns = ['연도', '건수']
                st.plotly_chart(px.line(trend, x='연도', y='건수', title=f"'{target_kwd}' 관련 지적사례 발생 추이"), use_container_width=True)
                
                # AI 리포트 생성 버튼
                if st.button("🚀 AI 종합 리포트 생성하기"):
                    with st.spinner("사례들을 종합하여 분석 보고서를 작성 중입니다..."):
                        try:
                            # 프롬프트에 넣을 사례 텍스트 생성 (최대 20개)
                            cases_summary = ""
                            for i, r in target_df.head(20).iterrows():
                                cases_summary += f"- [{r['결정연도']}] {r['회사명']} ({r['관련계정과목']}): {r['지적사항요약']}\n"
                            
                            prompt = f"""
                            당신은 회계법인의 품질관리실(Quality Control) 파트너입니다.
                            아래 제공된 **'{target_kwd}'** 관련 과거 감리지적사례들을 종합 분석하여 주니어 회계사들을 위한 교육용 리포트를 작성하세요.

                            [분석 대상 사례 목록]
                            {cases_summary}

                            [리포트 목차 및 요구사항]
                            1. **Risk Overview**: 해당 이슈({target_kwd})가 회계감사에서 왜 위험한지, 어떤 특징이 있는지 요약.
                            2. **Common Fraud Schemes**: 사례들에서 공통적으로 발견되는 회계부정/오류 수법 (구체적으로).
                            3. **Key Audit Procedures**: 감사인이 이를 적발하기 위해 반드시 수행해야 할 감사절차(Checklist) 5가지.
                            4. **Lesson Learned**: 결론 및 제언.

                            * 톤앤매너: 전문적이고 논리적으로 작성할 것.
                            * 중요 키워드는 굵게 표시할 것.
                            """
                            
                            response = model.generate_content(prompt).text
                            st.markdown(response)
                            save_ai_log(f"통합리포트: {target_kwd}", response)
                            
                        except Exception as e:
                            st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
            else:
                st.warning("해당 키워드로 검색된 사례가 없습니다. 다른 단어로 시도해보세요.")

    # [오른쪽] 기준서 챗봇 (기존 기능 유지)
    with col_bot:
        st.markdown("### 📘 기준서/감사기준 챗봇")
        st.info("공부하다 궁금한 기준서 내용을 물어보세요.")
        
        std_type = st.radio("검색 대상", ["전체 통합", "K-IFRS (회계기준)", "KGAAS (감사기준)"])
        use_google = st.toggle("Google 검색 연동", value=True, help="체크 시 최신 기준서를 구글링하여 답변합니다.")
        
        user_q = st.text_input("질문 입력", placeholder="예: 재고자산 실사 입회 생략 요건은?")
        
        if user_q:
            with st.spinner("기준서를 찾아보는 중..."):
                try:
                    # 페르소나 설정
                    if std_type == "K-IFRS (회계기준)":
                        persona = "당신은 K-IFRS(한국채택국제회계기준) 전문 위원입니다."
                        query_prefix = "K-IFRS"
                    elif std_type == "KGAAS (감사기준)":
                        persona = "당신은 회계감사기준(KGAAS) 전문 위원입니다."
                        query_prefix = "회계감사기준"
                    else:
                        persona = "당신은 회계 및 감사기준 통합 전문가입니다."
                        query_prefix = "K-IFRS 및 회계감사기준"

                    prompt = f"""
                    {persona}
                    사용자 질문: {user_q}
                    
                    [답변 원칙]
                    1. 반드시 **관련 기준서 번호(제1XXX호)**와 **문단 번호**를 명시하여 근거를 대세요.
                    2. 블로그나 뇌피셜이 아닌, 기준서 원문에 입각하여 정확하게 설명하세요.
                    """
                    
                    if use_google:
                        # 도구 재설정 (검색용)
                        search_model = genai.GenerativeModel(target_model, tools=[{"google_search": {}}])
                        final_prompt = f"Google 검색 키워드: '{query_prefix} {user_q}'\n{prompt}"
                        res = search_model.generate_content(final_prompt).text
                    else:
                        # 일반 생성
                        res = model.generate_content(prompt).text
                    
                    st.markdown(res)
                    save_ai_log(f"챗봇({std_type}): {user_q}", res)
                    
                except Exception as e:
                    st.error(f"답변 생성 실패: {e}")

