# Real-time-Object-Detection
유튜브 실시간 cctv 영상에서 이동 차량 수와 유동 인구를 파악할 수 있다.

<div align="center">
<img src="https://github.com/user-attachments/assets/be4a8946-3e2c-469e-8a20-8ddd5805952e"width=550>
<p>시연 영상</p>
</div>

--- 

## 실행

### 종속성 설치

필요한 python 종속성을 전부 설치: ```
pip install -r requirements.txt ```

### streamlink을 사용하여 유튜브 실시간 비디오 재생

cmd창을 열어 streamlink 설치: ```
pip install streamlink ```   
   

원하는 유튜브 실시간 영상의 url을 사용해 비디오 재생 (아래의 링크는 권장 유튜브 라이브 영상입니다.
```
streamlink "https://www.youtube.com/live/6dp-bvQ7RWo?si=VjDSk9v3SOEayKM5" best --hls-live-restart --player-external-http
```

### streamlink 로컬 주소 copy

비디오 재생시 재공되는 로컬 주소를 복사한다:

```
http://127.0.0.1:?????/
```

### 프로그램 실행

복사한 로컬 주소를 매개 변수에 넣어 python 파일을 실행시킨다:
```
python object_detection.py http://127.0.0.1:?????/
```

---

## 유의사항

- 실시간 영상이기 때문에 버퍼링이 있을 수 있습니다
- 모델 특성상 멀리 있거나 작은 객체는 제대로 인식하지 못할 수 있습니다
---
