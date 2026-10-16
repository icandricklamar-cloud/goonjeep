import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 브라우저 탭 제목 및 아이콘 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 화면 상단 제목
st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    return pd.read_csv(url, encoding="utf-8")

raw_df = load_data()
total_count = len(raw_df)

# 3. 전처리 및 속성 생성
required_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi', 'movieNm']
df = raw_df.dropna(subset=required_cols).copy()
df = df[df['first_week_audi'] > 0].copy()

# 파생 속성 생성
df['스크린 수 (log)'] = np.log10(df['first_scrn'])
df['누적 관객 (log)'] = np.log10(df['total_audi'])
df['10위권 일수'] = df['days_in_top10']
df['롱런 지수'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20)

# 원래 단위 분석용 컬럼
df['스크린 수'] = df['first_scrn']
df['누적 관객'] = df['total_audi']

clustered_count = len(df)

# 전체 편수와 묶은 편수 한 줄 표시
st.write(f"전체 영화 편수: **{total_count}**편 | 묶은 영화 편수: **{clustered_count}**편")
st.markdown("---")

# 4. 묶는 데 사용할 속성 선택 (다중 선택, 기본값 4개)
feature_options = ['스크린 수 (log)', '누적 관객 (log)', '10위권 일수', '롱런 지수']
selected_features = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=feature_options,
    default=feature_options
)

if len(selected_features) < 2:
    st.warning("군집화를 진행하려면 속성을 2개 이상 선택해 주세요.")
    st.stop()

# 5. 표준화 및 k-평균 군집화 (난수 고정 random_state=42)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 부여
avg_audi = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
label_map = {raw_id: label for raw_id, label in zip(avg_audi.index, ['㉮', '㉯', '㉰'])}
df['묶음'] = df['cluster_raw'].map(label_map)

cluster_order = ['㉮', '㉯', '㉰']

# 6. 2차원 산점도
st.markdown("### 📊 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_2d = st.selectbox("2차원 가로축 속성:", options=selected_features, index=0)
with col2:
    y_2d = st.selectbox("2차원 세로축 속성:", options=selected_features, index=min(1, len(selected_features)-1))

fig_2d = px.scatter(
    df,
    x=x_2d,
    y=y_2d,
    color='묶음',
    hover_name='movieNm',
    category_orders={'묶음': cluster_order},
    title=f"2차원 산점도 ({x_2d} vs {y_2d})"
)
fig_2d.update_traces(marker=dict(size=6))
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("---")

# 7. 3차원 산점도 (선택 속성이 3개 미만이면 안내 문구 표시)
st.markdown("### 🎲 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 표시하려면 선택한 속성이 3개 이상이어야 합니다.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_3d = st.selectbox("3차원 X축 속성:", options=selected_features, index=0)
    with col4:
        y_3d = st.selectbox("3차원 Y축 속성:", options=selected_features, index=1)
    with col5:
        z_3d = st.selectbox("3차원 Z축 속성:", options=selected_features, index=2)

    fig_3d = px.scatter_3d(
        df,
        x=x_3d,
        y=y_3d,
        z=z_3d,
        color='묶음',
        hover_name='movieNm',
        category_orders={'묶음': cluster_order},
        title=f"3차원 산점도 ({x_3d} vs {y_3d} vs {z_3d})"
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 8. 묶음별 편수 및 네 속성의 평균 (원래 단위)
st.markdown("### 📋 묶음별 편수 및 속성 평균 (원래 단위)")
summary_df = df.groupby('묶음').agg(
    편수=('movieNm', 'count'),
    스크린_수_평균=('스크린 수', 'mean'),
    누적_관객_평균=('누적 관객', 'mean'),
    10위권_일수_평균=('10위권 일수', 'mean'),
    롱런_지수_평균=('롱런 지수', 'mean')
).reindex(cluster_order).reset_index()

summary_df.columns = ['묶음', '편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']

# 보기 편하게 반올림 처리
summary_df['스크린 수 평균'] = summary_df['스크린 수 평균'].round(1)
summary_df['누적 관객 평균'] = summary_df['누적 관객 평균'].round(1)
summary_df['10위권 일수 평균'] = summary_df['10위권 일수 평균'].round(1)
summary_df['롱런 지수 평균'] = summary_df['롱런 지수 평균'].round(2)

st.dataframe(summary_df, use_container_width=True)

st.markdown("---")

# 9. 묶음별 누적 관객 상위 5개 영화 제목
st.markdown("### 🏆 묶음별 누적 관객 상위 5개 영화")
cols = st.columns(3)
for idx, c_label in enumerate(cluster_order):
    with cols[idx]:
        st.subheader(f"묶음 {c_label}")
        top5 = df[df['묶음'] == c_label].sort_values(by='total_audi', ascending=False).head(5)
        top5_display = top5[['movieNm', 'total_audi']].reset_index(drop=True)
        top5_display.columns = ['영화 제목', '누적 관객 수']
        top5_display['순위'] = range(1, len(top5_display) + 1)
        top5_display['누적 관객 수'] = top5_display['누적 관객 수'].apply(lambda x: f"{x:,}명")
        top5_display = top5_display[['순위', '영화 제목', '누적 관객 수']]
        st.table(top5_display)
