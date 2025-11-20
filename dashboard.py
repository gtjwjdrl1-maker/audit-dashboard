import streamlit as st
import pandas as pd
import sqlite3
import os
import google.generativeai as genai
import plotly.express as px 
from collections import Counter

# ==========================================
# 1. 설정 (API 키 입력 필수!)
# ==========================================
# 스트림릿 클라우드의 비밀 금고(Secrets)에서 키를 가져온다
try:
    MY_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # 로컬에서 테스트할 때는 그냥 빈 값 (또는 본인 키를 임시로 넣었다 지우기)
    MY_API_KEY = "API_KEY_MISSING"  
DB_FILE = "audit_database.db"

genai.configure(api_key=MY_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="회계감리 분석 시스템", layout="wide")

# ==========================================
# 2. 데이터 로드 및 그룹 매핑 (핵심!)
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM cases"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 1. 컬럼명 띄어쓰기 제거
    df.columns = [c.replace(' ', '') for c in df.columns]
    
    # 2. 연도 정제
    if '결정연도' in df.columns:
        df['결정연도'] = df['결정연도'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df = df[df['결정연도'] != '']
        df = df.sort_values('결정연도')

    # 3. [실무 그룹 매핑 엔진]
    def map_group(account_str):
        if pd.isna(account_str): return "📝 기타/주석"
        target = str(account_str).replace(" ", "")
        
        if any(x in target for x in ['매출', '수익', '채권', '미수', '대손']):
            return "💰 매출·채권 (Revenue)"
        elif any(x in target for x in ['재고', '매출원가', '매입', '채무']):
            return "📦 재고·매입 (Inventory)"
        elif any(x in target for x in ['금융', '주식', '파생', '투자', '현금', '예금', '대여']):
            return "🏦 금융·현금 (Financial)"
        elif any(x in target for x in ['유형', '무형', '감가', '손상', '부동산', '개발비', '영업권']):
            return "🏗️ 유·무형자산 (Assets)"
        elif any(x in target for x in ['연결', '지분법', '관계기업', '종속']):
            return "🔗 연결·지분법 (Consolidation)"
        elif any(x in target for x in ['자본', '잉여금', '주식보상', '자기주식']):
            return "💎 자본 (Equity)"
        elif any(x in target for x in ['법인세', '이연']):
            return "⚖️ 법인세 (Tax)"
        else:
            return "📝 기타/주석 (Others)"

    df['표준그룹'] = df['관련계정과목'].apply(map_group)
    return df

df_all = load_data()

if df_all.empty:
    st.error("데이터가 없습니다.")
    st.stop()

# ==========================================
# 3. 메인 화면 구성
# ==========================================
st.title("📊 회계감리 지적사례 AI 분석 시스템")
st.caption("금감원 지적사례를 실무 관점(Cycle)으로 재분류하여 분석합니다.")

tab1, tab2 = st.tabs(["1️⃣ 종합 개요 (Overview)", "2️⃣ 계정별 심화 분석 (Deep Dive)"])

# --------------------------------------------------------------------
# [탭 1] 종합 개요: 시각화 & 검색 포털
# --------------------------------------------------------------------
with tab1:
    # 1. KPI 지표
    total_files = len(df_all)
    top_group = df_all['표준그룹'].mode()[0]
    top_violation = df_all['위반유형'].value_counts().idxmax() if '위반유형' in df_all.columns else "-"

    col1, col2, col3 = st.columns(3)
    col1.metric("총 분석 파일 수", f"{total_files} 개", "Audit Cases")
    col2.metric("최다 적발 영역", top_group.split(' ')[1], "Risk High")
    col3.metric("최다 위반 유형", top_violation, "Caution")

    st.markdown("---")

    # 2. 차트 영역 (원형 차트 + 트렌드)
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🎨 실무 그룹별 지적 비중")
        group_counts = df_all['표준그룹'].value_counts().reset_index()
        group_counts.columns = ['그룹', '건수']
        # [원형 차트] 도넛 형태로 깔끔하게
        fig_pie = px.pie(group_counts, values='건수', names='그룹', hole=0.4)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("📈 연도별 지적 추이")
        if '결정연도' in df_all.columns:
            year_counts = df_all['결정연도'].value_counts().sort_index().reset_index()
            year_counts.columns = ['연도', '건수']
            fig_line = px.line(year_counts, x='연도', y='건수', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # 3. 검색 및 상세 보기 (Drill-down)
    st.subheader("🔎 사례 검색 및 상세 조회")
    
    col_search, col_result = st.columns([1, 2])
    
    with col_search:
        search_keyword = st.text_input("키워드 입력", placeholder="예: 횡령, 주석, 파생상품")
        
        # 검색 로직
        if search_keyword:
            mask = df_all.apply(lambda x: x.astype(str).str.contains(search_keyword).any(), axis=1)
            search_results = df_all[mask]
        else:
            search_results = df_all
            
        st.caption(f"검색 결과: {len(search_results)} 건")
        
        # 라벨 생성: [그룹] 회사명 - 계정
        search_results['Label'] = "[" + search_results['표준그룹'] + "] " + search_results['회사명'] + " - " + search_results['관련계정과목']
        
        # 선택 박스
        selected_case_label = st.selectbox("결과 목록 (선택하세요)", search_results['Label'].unique())

    with col_result:
        if selected_case_label:
            # 선택된 행 찾기
            row = search_results[search_results['Label'] == selected_case_label].iloc[0]
            
            # 상세 카드 디자인
            st.info(f"📄 **파일명:** {row['파일명']}  |  🏢 **회사명:** {row['회사명']} ({row['결정연도']})")
            
            with st.container(border=True):
                st.markdown("#### ⚠️ 지적 사항 요약")
                st.write(row['지적사항요약'])
                
            with st.container(border=True):
                st.markdown("#### 💡 감사인 유의사항 (AI 분석)")
                st.success(row['감사인유의사항'])

# --------------------------------------------------------------------
# [탭 2] 심화 분석 (그룹별 통합 리포트)
# --------------------------------------------------------------------
with tab2:
    st.markdown("### 🤖 계정 그룹별 AI 인사이트")
    
    target_group = st.selectbox("분석할 업무 영역(Cycle) 선택", sorted(df_all['표준그룹'].unique()))
    group_df = df_all[df_all['표준그룹'] == target_group]
    
    st.success(f"👉 **'{target_group}'** 영역에서 총 **{len(group_df)}건**의 사례가 발견되었습니다.")
    
    # 통계 차트
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📆 연도별 추이**")
        trend = group_df['결정연도'].value_counts().sort_index().reset_index()
        trend.columns = ['연도', '건수']
        st.plotly_chart(px.line(trend, x='연도', y='건수', markers=True), use_container_width=True)
    with c2:
        st.markdown("**🚨 주요 위반 유형**")
        if '위반유형' in group_df.columns:
            v_counts = group_df['위반유형'].value_counts().head(5).reset_index()
            v_counts.columns = ['유형', '건수']
            st.plotly_chart(px.pie(v_counts, values='건수', names='유형', hole=0.4), use_container_width=True)

    st.markdown("---")
    if st.button("🚀 AI 실전 가이드 생성"):
        with st.spinner("AI가 사례를 분석 중입니다..."):
            try:
                stats_summary = f"총 {len(group_df)}건. 최빈 유형: {group_df['위반유형'].mode()[0] if '위반유형' in group_df else '미상'}"
                cases_text = ""
                for idx, row in group_df.sort_values('결정연도', ascending=False).head(30).iterrows():
                     cases_text += f"- [{row['결정연도']}] {row['회사명']} ({row.get('위반유형','-')}): {row['지적사항요약']}\n"

                prompt = f"""
                당신은 회계법인 파트너입니다. '{target_group}' 영역의 과거 지적 사례를 분석하여 감사팀 교육 자료를 작성하세요.
                
                [데이터] {stats_summary}
                [사례] {cases_text[:15000]}

                보고서 목차:
                1. **Risk Overview**: 해당 계정 그룹의 주요 부정 패턴 요약
                2. **Key Case Study**: 가장 빈번하거나 치명적인 위반 사례의 수법 분석
                3. **Audit Action Plan 5**: 현장에서 반드시 수행해야 할 구체적 감사 절차 5가지 (명령조로 작성)
                """
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"오류: {e}")