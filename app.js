// DOM이 모두 로드된 후에 스크립트를 실행합니다.
document.addEventListener('DOMContentLoaded', () => {

    // DOM 요소 선택
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('fileInput');
    const videoListContainer = document.getElementById('videoListContainer');
    const videoPlayer = document.getElementById('videoPlayer');
    const playerPlaceholder = document.getElementById('playerPlaceholder');
    const analysisResult = document.getElementById('analysisResult');
    const lieDetectorButton = document.getElementById('lieDetectorButton');

    const editControlsWrapper = document.getElementById('editControlsWrapper');
    const timelineScrubberContainer = document.getElementById('timelineScrubberContainer');
    const segmentControls = document.getElementById('segmentControls');
    const timelineScrubber = document.getElementById('timelineScrubber');
    const startHandle = document.getElementById('startHandle');
    const endHandle = document.getElementById('endHandle');
    const selectedRange = document.getElementById('selectedRange');
    const currentTimeMarker = document.getElementById('currentTimeMarker');
    const startTimeLabel = document.getElementById('startTimeLabel');
    const endTimeLabel = document.getElementById('endTimeLabel');
    const playSegmentBtn = document.getElementById('playSegmentBtn');
    const timeDisplay = document.getElementById('timeDisplay');

    // [서버 연동] 현재 선택된 비디오의 원본 File 객체를 저장할 변수
    let currentSelectedFile = null; 

    // 상태 변수
    let editStartTime = 0.0;
    let editEndTime = 0.0;
    let isDraggingStart = false;
    let isDraggingEnd = false;
    let scrubberWidth = 0;
    let videoDuration = 0;

    const MAX_EDIT_DURATION = 60; // 최대 편집 가능 시간 (60초)

    // --- 이벤트 리스너 ---

    // [서버 연동] 거짓말 탐지 버튼 클릭 시 서버로 요청
    lieDetectorButton.addEventListener('click', async () => {
        // 1. 현재 선택된 파일과 편집 구간이 유효한지 확인
        if (!currentSelectedFile) {
            alert("먼저 왼쪽 목록에서 비디오를 선택해주세요.");
            return;
        }
        if (editEndTime - editStartTime <= 0) {
            alert("유효한 편집 구간을 설정해주세요.");
            return;
        }

        // 2. 로딩 상태 표시
        analysisResult.innerHTML = "<h3>분석 중...</h3><p>서버에서 모델을 실행하고 있습니다. 잠시 기다려주세요...</p>";

        // 3. FormData 객체 생성 (파일과 데이터를 함께 전송)
        const formData = new FormData();
        formData.append('video', currentSelectedFile); // 비디오 원본 파일
        formData.append('startTime', editStartTime.toString()); // 시작 시간
        formData.append('endTime', editEndTime.toString());   // 종료 시간

        try {
            // 4. Flask 서버(http://127.0.0.1:5000/analyze)로 POST 요청 전송
            const response = await fetch('http://127.0.0.1:5000/analyze', {
                method: 'POST',
                body: formData, // 폼 데이터 전송
            });

            if (!response.ok) {
                // 서버가 4xx, 5xx 에러를 반환한 경우
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with ${response.status}`);
            }

            // 5. 서버로부터 성공적인 응답(JSON) 받기
            const data = await response.json();

            // 6. 우측 하단 패널에 결과 표시
            analysisResult.innerHTML = `
                <p><strong>결과:</strong> <span style="color: ${data.result === '거짓 (Lie)' ? '#dc3545' : '#28a745'}; font-weight: bold;">${data.result}</span></p>
                <p><strong>정확도:</strong> ${data.accuracy.toFixed(2)} %</p>
            `;

        } catch (error) {
            // 7. 네트워크 오류 또는 서버 처리 오류 시
            console.error("Analysis Error:", error);
            analysisResult.innerHTML = `
                <h3>분석 실패</h3>
                <p style="color: #dc3545;">오류가 발생했습니다: ${error.message}</p>
            `;
        }
    });

    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
        handleFiles(event.target.files);
    });

    // 드래그 앤 드롭 이벤트
    videoListContainer.addEventListener('dragover', (event) => {
        event.preventDefault();
        videoListContainer.classList.add('dragover');
    });

    videoListContainer.addEventListener('dragleave', () => {
        videoListContainer.classList.remove('dragover');
    });

    videoListContainer.addEventListener('drop', (event) => {
        event.preventDefault();
        videoListContainer.classList.remove('dragover');
        handleFiles(event.dataTransfer.files);
    });

    // 비디오 플레이어 이벤트
    videoPlayer.addEventListener('loadedmetadata', () => {
        videoDuration = videoPlayer.duration;
        editControlsWrapper.style.display = 'block';
        timelineScrubberContainer.style.display = 'flex';
        segmentControls.style.display = 'flex';

        generateScrubberThumbnails();

        editStartTime = 0;
        // 1분 제한보다 영상 길이가 짧으면, 끝 지점을 영상 길이로 설정
        if (videoDuration < MAX_EDIT_DURATION) {
            editEndTime = videoDuration;
        } else {
            editEndTime = MAX_EDIT_DURATION; // 영상이 길면 기본 1분
        }
        scrubberWidth = timelineScrubber.offsetWidth;
        updateScrubberUI();
    });

    videoPlayer.addEventListener('timeupdate', () => {
        if (!isDraggingStart && !isDraggingEnd) {
            const progress = videoPlayer.currentTime / videoDuration;
            if (scrubberWidth > 0) {
                currentTimeMarker.style.left = `${progress * scrubberWidth}px`;
            }
        }
    });

    videoPlayer.addEventListener('canplay', () => {
        updateScrubberUI();
    });

    videoPlayer.addEventListener('pause', () => {
        videoPlayer.removeEventListener('timeupdate', segmentPlayHandler);
    });

    // 스크러버(타임라인) 핸들 드래그 이벤트
    startHandle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDraggingStart = true;
    });

    endHandle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDraggingEnd = true;
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDraggingStart && !isDraggingEnd) return;
        scrubberWidth = timelineScrubber.offsetWidth;

        const mouseX = getMousePositionInScrubber(e);
        let newTime = (mouseX / scrubberWidth) * videoDuration;

        if (isDraggingStart) { // 시작 핸들 드래그
            newTime = Math.max(0, Math.min(newTime, editEndTime - (1 / videoDuration)));
            editStartTime = newTime;

            if (editEndTime - editStartTime > MAX_EDIT_DURATION) {
                editEndTime = editStartTime + MAX_EDIT_DURATION;
            }
            
            if (editEndTime > videoDuration) {
                editEndTime = videoDuration;
                editStartTime = Math.max(0, videoDuration - MAX_EDIT_DURATION);
            }
            
        } else if (isDraggingEnd) { // 종료 핸들 드래그
            newTime = Math.min(videoDuration, Math.max(newTime, editStartTime + (1 / videoDuration)));
            editEndTime = newTime;

            if (editEndTime - editStartTime > MAX_EDIT_DURATION) {
                editStartTime = editEndTime - MAX_EDIT_DURATION;
            }

            if (editStartTime < 0) {
                editStartTime = 0;
                editEndTime = Math.min(MAX_EDIT_DURATION, videoDuration);
            }
        }
        
        if (editStartTime < 0) editStartTime = 0;
        if (editEndTime > videoDuration) editEndTime = videoDuration;

        updateScrubberUI();
        videoPlayer.currentTime = newTime;
    });

    document.addEventListener('mouseup', () => {
        isDraggingStart = false;
        isDraggingEnd = false;
    });

    // 스크러버 클릭 시 비디오 시간 이동
    timelineScrubber.addEventListener('click', (e) => {
        if (e.target === startHandle || e.target === endHandle || e.target.classList.contains('time-label')) {
            return;
        }
        const mouseX = getMousePositionInScrubber(e);
        const clickTime = (mouseX / scrubberWidth) * videoDuration;
        videoPlayer.currentTime = clickTime;
    });

    // 선택 구간 재생 버튼
    playSegmentBtn.addEventListener('click', () => {
        videoPlayer.removeEventListener('timeupdate', segmentPlayHandler);
        
        videoPlayer.currentTime = editStartTime;
        videoPlayer.play();
        
        videoPlayer.addEventListener('timeupdate', segmentPlayHandler);
    });

    // --- 헬퍼 함수 ---

    /** 파일 처리 */
    function handleFiles(files) {
        const placeholder = videoListContainer.querySelector('.video-list-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
        for (const file of files) {
            if (file.type.startsWith('video/')) {
                addVideoToList(file);
            }
        }
    }

// app.js 에서
// (다른 코드는 다 그대로 두시고, 이 함수만 교체하세요)

/** [최종 수정 2] 비디오 목록에 아이템 추가 및 썸네일 생성 */
function addVideoToList(file) {
    // 1. 파일 객체로부터 메모리 URL 생성 (메인 플레이어와 공유)
    const videoURL = URL.createObjectURL(file);

    // 2. 썸네일 아이템 (div) 생성 (스타일 태그에 배경색 추가)
    const thumbnailDiv = document.createElement('div');
    thumbnailDiv.className = 'video-thumbnail';
    thumbnailDiv.innerHTML = `
        <img src="" alt="Thumbnail" style="width: 160px; height: 90px; object-fit: cover; background-color: #222;">
        <div class="video-thumbnail-name">${file.name}</div>
    `;
    const thumbnailImg = thumbnailDiv.querySelector('img');

    // 3. 썸네일 생성용 임시 <video> 태그 생성
    const tempVideo = document.createElement('video');
    tempVideo.muted = true;
    tempVideo.style.display = 'none'; // 화면에는 보이지 않게 처리

    // 4. [이벤트 1] '탐색(seek)'이 '완료'되면 캔버스에 그리고 비디오 삭제 (최종 단계)
    const onSeeked = () => {
        
        // ⬇️ [핵심 수정] 
        // seeked 이벤트가 발생해도, 프레임이 렌더링될 시간을 100ms 정도 더 줍니다.
        setTimeout(() => {
            const canvas = document.createElement('canvas');
            canvas.width = 160;
            canvas.height = 90;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(tempVideo, 0, 0, canvas.width, canvas.height);
            
            thumbnailImg.src = canvas.toDataURL('image/jpeg');
            
            // [중요] 작업 완료 후 DOM에서 임시 비디오 요소를 반드시 제거
            if (document.body.contains(tempVideo)) {
                document.body.removeChild(tempVideo);
            }
        }, 100); // 100ms (0.1초) 지연
    };

    // 5. [이벤트 2] 비디오 '데이터'(길이 등)가 '로드'되면, 탐색 시작
    const onLoadedData = () => {
        const seekTime = (tempVideo.duration > 1) ? 1.0 : (tempVideo.duration / 2);
        tempVideo.currentTime = seekTime; // ⬅️ 이 코드가 'seeked' 이벤트를 유발
    };
    
    // 6. [이벤트 3] 비디오 로드 '오류' 처리
    const onError = () => {
        console.error("썸네일 생성 실패 (비디오 로드 오류):", file.name);
        if (document.body.contains(tempVideo)) {
            document.body.removeChild(tempVideo);
        }
    };

    // 7. 이벤트 리스너를 *먼저* 등록 (1회만 실행되도록 { once: true })
    tempVideo.addEventListener('loadeddata', onLoadedData, { once: true });
    tempVideo.addEventListener('seeked', onSeeked, { once: true });
    tempVideo.addEventListener('error', onError, { once: true });

    // 8. [순서 중요] DOM에 추가 *먼저* 하고,
    document.body.appendChild(tempVideo);
    // 9. *그 다음에* 소스를 설정하여 로드 시작
    tempVideo.src = videoURL;

    // 10. 썸네일 클릭 이벤트 (메인 플레이어 영상 교체)
    thumbnailDiv.addEventListener('click', () => {
        playerPlaceholder.style.display = 'none';
        videoPlayer.style.display = 'block';
        videoPlayer.src = videoURL; 
        analysisResult.innerText = "Select a video and drag the handles to set an edit range.";
        currentSelectedFile = file; 
        clipListContainer.innerHTML = "<p class='clip-list-placeholder'>Create a list by pressing the clip button.</p>";
    });
    

    // 11. 완성된 썸네일 아이템을 목록에 추가
    videoListContainer.appendChild(thumbnailDiv);
}
    /** 타임라인에 썸네일 스트립 생성 */
    function generateScrubberThumbnails() {
        // 기존 썸네일 및 핸들 외 요소 제거
        timelineScrubber.innerHTML = '';
        timelineScrubber.appendChild(selectedRange);
        timelineScrubber.appendChild(startHandle);
        timelineScrubber.appendChild(endHandle);
        timelineScrubber.appendChild(currentTimeMarker);

        const segmentDuration = 2; // 2초 간격 썸네일
        scrubberWidth = timelineScrubber.offsetWidth;

        const thumbVideo = document.createElement('video');
        thumbVideo.src = videoPlayer.src;
        thumbVideo.crossOrigin = "anonymous";
        thumbVideo.muted = true;
        thumbVideo.style.display = 'none';

        document.body.appendChild(thumbVideo);

        let currentThumbIndex = 0;

        const captureThumbnail = () => {
            const time = currentThumbIndex * segmentDuration;
            if (time > videoDuration) {
                if (document.body.contains(thumbVideo)) {
                    document.body.removeChild(thumbVideo);
                }
                return;
            }

            thumbVideo.currentTime = time;
            thumbVideo.onseeked = () => {
                const canvas = document.createElement('canvas');
                canvas.width = thumbVideo.videoWidth;
                canvas.height = thumbVideo.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(thumbVideo, 0, 0, canvas.width, canvas.height);

                const img = document.createElement('img');
                img.src = canvas.toDataURL();
                img.className = 'timeline-scrubber-thumbnail';
                
                const thumbWidth = (segmentDuration / videoDuration) * scrubberWidth;
                img.style.width = `${Math.max(1, thumbWidth)}px`;
                timelineScrubber.insertBefore(img, selectedRange);

                currentThumbIndex++;
                captureThumbnail();
            };
        };
        
        thumbVideo.addEventListener('loadedmetadata', captureThumbnail, { once: true });
    }

    /** 스크러버 내 마우스 X좌표 계산 */
    function getMousePositionInScrubber(event) {
        const rect = timelineScrubber.getBoundingClientRect();
        let x = event.clientX - rect.left;
        x = Math.max(0, Math.min(x, scrubberWidth));
        return x;
    }

    /** 선택 구간 재생 핸들러 */
    const segmentPlayHandler = () => {
        if (videoPlayer.currentTime >= editEndTime) {
            videoPlayer.pause();
            videoPlayer.removeEventListener('timeupdate', segmentPlayHandler);
        }
    };

    /** 스크러버 UI 업데이트 */
    function updateScrubberUI() {
        const currentScrubberWidth = timelineScrubber.offsetWidth;
        
        if (!videoDuration || currentScrubberWidth === 0) {
            return;
        }
        
        scrubberWidth = currentScrubberWidth;

        const startPercent = (editStartTime / videoDuration);
        const endPercent = (editEndTime / videoDuration);

        startHandle.style.left = `${startPercent * scrubberWidth}px`;
        endHandle.style.left = `${endPercent * scrubberWidth}px`; 

        selectedRange.style.left = `${startPercent * scrubberWidth}px`;
        selectedRange.style.width = `${(endPercent - startPercent) * scrubberWidth}px`;

        const startTimeFormatted = formatTime(editStartTime);
        const endTimeFormatted = formatTime(editEndTime);
        startTimeLabel.innerText = startTimeFormatted;
        endTimeLabel.innerText = endTimeFormatted;
        
        timeDisplay.innerText = `Selected: ${startTimeFormatted} - ${endTimeFormatted}`;
    }

    /** 시간 포맷팅 (mm:ss.ms) */
    function formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) seconds = 0;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        const ms = Math.floor((seconds - Math.floor(seconds)) * 100);

        const formattedMinutes = String(minutes).padStart(2, '0');
        const formattedSeconds = String(remainingSeconds).padStart(2, '0');
        const formattedMs = String(ms).padStart(2, '0');

        return `${formattedMinutes}:${formattedSeconds}.${formattedMs}`;
    }

    // 창 크기 조절 시 스크러버 UI 업데이트
    new ResizeObserver(() => {
        updateScrubberUI();
    }).observe(timelineScrubber);

    // ---------------------------------------------------------
// Whisper 기반 "영상 분석" 기능 추가
// ---------------------------------------------------------

const analyzeButton = document.getElementById('analyzeButton');       // HTML에 버튼(id="analyzeButton")
const clipListContainer = document.getElementById('clipListContainer'); // 오른쪽 사이드(id="clipListContainer")

if (analyzeButton) {
    analyzeButton.addEventListener('click', async () => {
        if (!currentSelectedFile) {
            alert("Please choose the video first");
            return;
        }

        clipListContainer.innerHTML = "<p class='clip-list-placeholder'>It takes some time...</p>";

        const formData = new FormData();
        formData.append('video', currentSelectedFile);

        try {
            const response = await fetch('http://127.0.0.1:5000/analyze_full', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `Server responded with ${response.status}`);
            }

            const segments = await response.json();
            if (!segments.length) {
                clipListContainer.innerHTML = "<p class='clip-list-placeholder'>Sentence section not found.</p>";
                return;
            }

            clipListContainer.innerHTML = "";
            

// 각 문장별 썸네일 + 시간 + 텍스트 표시
segments.forEach((seg, i) => {
    const item = document.createElement('div');
    item.className = 'clip-item';

    // 썸네일
    const img = document.createElement('img');
    img.className = 'clip-thumb';
    img.alt = `Segment ${i + 1}`;
    img.src = ''; // 썸네일 나중에 채워짐

    // 시간
    const timeLabel = document.createElement('div');
    timeLabel.className = 'clip-time';
    timeLabel.textContent = `${formatTime(seg.start)} - ${formatTime(seg.end)}`;

    // 텍스트
    const caption = document.createElement('div');
    caption.className = 'clip-caption';
    caption.textContent = seg.text;

    // 클릭 시 타임라인 구간 이동
    item.addEventListener('click', () => {
        editStartTime = seg.start;
        editEndTime = seg.end;
        updateScrubberUI();
        videoPlayer.currentTime = editStartTime;
    });

    item.appendChild(img);
    item.appendChild(timeLabel);
    item.appendChild(caption);
    clipListContainer.appendChild(item);

    // 🔹 비디오 프레임에서 썸네일 생성
    const tempVideo = document.createElement('video');
    tempVideo.src = videoPlayer.src;
    tempVideo.muted = true;
    tempVideo.crossOrigin = "anonymous";
    tempVideo.addEventListener('loadedmetadata', () => {
        tempVideo.currentTime = Math.min(seg.start + 0.3, seg.end); // 구간 초반부 프레임
    });
    tempVideo.addEventListener('seeked', () => {
        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 90;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(tempVideo, 0, 0, canvas.width, canvas.height);
        img.src = canvas.toDataURL('image/jpeg');
    });
});


        } catch (error) {
            clipListContainer.innerHTML = `<p class='clip-list-placeholder' style='color:#dc3545;'>오류 발생: ${error.message}</p>`;
        }
    });
}


}); // 'DOMContentLoaded' 리스너 종료