from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.decomposition import PCA

app = FastAPI()

# 허용할 도메인 설정
origins = [
    "http://localhost:8085",  # SpringBoot 서버 도메인
    # 필요한 다른 도메인을 추가
]

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 도메인
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.get("/")
async def read_root():
    return {"message": "Hello World!"}

@app.post("/api/statistic/clustering/{n_clusters}")
async def kmeansClustering(n_clusters: int, request: Request):
    req = await request.json() # 요청 데이터 JSON으로 받기
    data = pd.DataFrame(req)
    print(data) # 서버 콘솔에 출력

    # 텍스트 전처리
    answers = data['answerContent'].astype(str).tolist() # 답변 컬럼을 리스트로 변환

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(answers)

    print(X)

    # 군집 개수 설정
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # 군집 레이블 추가
    data['Cluster'] = kmeans.labels_
    print(data[['answerContent', 'Cluster']].head())

    # PCA로 차원 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # PCA 결과를 데이터프레임으로 변환
    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = kmeans.labels_

    print(df_pca)

    return {
        "message" : "Data received successfully",
        "data" : df_pca.to_dict(orient='records') # DataFrame을 JSON으로 변환
        }