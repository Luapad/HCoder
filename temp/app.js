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
    
    // [팝업 추가] 모달 DOM 요소 선택
    const resultModal = document.getElementById('resultModal');
    const modalBodyContent = document.getElementById('modalBodyContent');
    const modalCloseBtn = document.querySelector('.modal-close-btn');

    // [서버 연동] 현재 선택된 비디오의 원본 File 객체를 저장할 변수
    let currentSelectedFile = null;
    
    // [유지] 파싱 버튼 로직 유지를 위해 allSegments 변수는 그대로 둡니다.
    let allSegments = []; 

    // 상태 변수
    let editStartTime = 0.0;
    let editEndTime = 0.0;
    let isDraggingStart = false;
    let isDraggingEnd = false;
    let scrubberWidth = 0;
    let videoDuration = 0;

    const MAX_EDIT_DURATION = 60; // 최대 편집 가능 시간 (60초)

    // --- 이벤트 리스너 ---

    // [수정] 거짓말 탐지 버튼 클릭 시 (B 방식: startTime/endTime 전송)
    lieDetectorButton.addEventListener('click', async () => {
        if (!currentSelectedFile) {
            alert("먼저 왼쪽 목록에서 비디오를 선택해주세요.");
            return;
        }

        if (editEndTime - editStartTime <= 0) {
            alert("유효한 편집 구간을 설정해주세요.");
            return;
        }

        // [수정] B 방식 경고 메시지 추가
        analysisResult.innerHTML = "<h3>Analyzing...</h3><p>The model is running on the server. Please wait a moment.</p>";

        const formData = new FormData();
        formData.append('video', currentSelectedFile);
        
        // [수정] startTime, endTime을 전송합니다. (서버가 Whisper를 재실행하도록)
        formData.append('startTime', editStartTime.toString());
        formData.append('endTime', editEndTime.toString());

        try {
            const response = await fetch('http://127.0.0.1:5000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with ${response.status}`);
            }

            const resultsList = await response.json(); 
            
            analysisResult.innerHTML = "";
            modalBodyContent.innerHTML = "";

            if (resultsList.length > 0) {
                resultsList.forEach(item => {
                    const resultColor = item.result === 'Lie' ? '#dc3545' : '#28a745';

                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'segment-result-item';
                    itemDiv.style.borderBottom = '1px solid #444';
                    itemDiv.style.padding = '10px 0';

                    itemDiv.innerHTML = `
                        <p style="margin: 0 0 5px 0;">
                            <strong style="color: ${resultColor}; font-size: 1.1em;">[ ${item.result.toUpperCase()} ]</strong>
                            (${item.accuracy.toFixed(2)}% confidence)
                        </p>
                        <p style="margin: 0 0 5px 0; font-size: 0.9em; color: #aaa;">
                            (${formatTime(item.start)} - ${formatTime(item.end)})
                        </p>
                        <p style="margin: 0; font-style: italic; color: #eee;">
                            "${item.text}"
                        </p>
                    `;

                    itemDiv.addEventListener('click', () => {
                        videoPlayer.currentTime = item.start;
                    });
                    itemDiv.style.cursor = 'pointer'; 
                    itemDiv.title = 'Click to jump to this segment';

                    const modalItemDiv = itemDiv.cloneNode(true);
                    modalItemDiv.style.cursor = 'default'; 
                    modalItemDiv.title = '';

                    analysisResult.appendChild(itemDiv);
                    modalBodyContent.appendChild(modalItemDiv);
                });

                resultModal.style.display = 'block';

            } else {
                const noResultHtml = `<h3>분석 결과 없음</h3><p>선택하신 구간에 유효한 문장이 없거나 모델이 분석하지 못했습니다.</p>`;
                analysisResult.innerHTML = noResultHtml;
                modalBodyContent.innerHTML = noResultHtml;
            }

        } catch (error) {
            console.error("Analysis Error:", error);
            const errorHtml = `
                <h3>분석 실패</h3>
                <p style="color: #dc3545;">오류가 발생했습니다: ${error.message}</p>
            `;
            analysisResult.innerHTML = errorHtml;
            modalBodyContent.innerHTML = errorHtml;
        }
    });
    
    // [유지] analyzeButton 이벤트 리스너 (파싱 버튼 기능)는 그대로 유지됩니다.
    const clipListContainer = document.getElementById('clipListContainer');
    const analyzeButton = document.getElementById('analyzeButton');
    const segmentListTitle = document.getElementById('segmentListTitle');
    
    analyzeButton.addEventListener('click', async () => {
        if (!currentSelectedFile) {
            alert("먼저 왼쪽 목록에서 비디오를 선택해주세요.");
            return;
        }

        clipListContainer.innerHTML = "<h3>Parsing...</h3><p>The model is running on the server. Please wait a moment.</p>";
        

        const formData = new FormData();
        formData.append('video', currentSelectedFile);

        try {
            const response = await fetch('http://127.0.0.1:5000/analyze_full', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with ${response.status}`);
            }

            allSegments = await response.json(); 
            
            clipListContainer.innerHTML = "";
            segmentListTitle.innerText = "Sentence Segments";

            if (allSegments.length > 0) {
                allSegments.forEach(segment => {
                    const segmentDiv = document.createElement('div');
                    segmentDiv.className = 'segment-item';
                    segmentDiv.innerHTML = `
                        <p style="margin: 0; font-size: 0.9em; color: #aaa;">
                            (${formatTime(segment.start)} - ${formatTime(segment.end)})
                        </p>
                        <p style="margin: 5px 0 0 0; font-style: italic;">
                            "${segment.text}"
                        </p>
                    `;
                    segmentDiv.addEventListener('click', () => {
                        editStartTime = segment.start;
                        editEndTime = segment.end;
                        updateScrubberUI();
                        videoPlayer.currentTime = segment.start;
                    });
                    clipListContainer.appendChild(segmentDiv);
                });
            } else {
                clipListContainer.innerHTML = "<p class='clip-list-placeholder'>No spoken segments found.</p>";
            }


        } catch (error) {
            console.error("Parsing Error:", error);
            segmentListTitle.innerText = "Parsing Failed";
            clipListContainer.innerHTML = `<p style="color: #dc3545;">Parsing failed: ${error.message}</p>`;
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
        editEndTime = videoDuration < MAX_EDIT_DURATION ? videoDuration : MAX_EDIT_DURATION;

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

    // 스크러버 핸들 드래그
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

        if (isDraggingStart) {
            newTime = Math.max(0, Math.min(newTime, editEndTime - (1 / videoDuration)));
            editStartTime = newTime;
            if (editEndTime - editStartTime > MAX_EDIT_DURATION)
                editEndTime = editStartTime + MAX_EDIT_DURATION;
            if (editEndTime > videoDuration) {
                editEndTime = videoDuration;
                editStartTime = Math.max(0, videoDuration - MAX_EDIT_DURATION);
            }
        } else if (isDraggingEnd) {
            newTime = Math.min(videoDuration, Math.max(newTime, editStartTime + (1 / videoDuration)));
            editEndTime = newTime;
            if (editEndTime - editStartTime > MAX_EDIT_DURATION)
                editStartTime = editEndTime - MAX_EDIT_DURATION;
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

    timelineScrubber.addEventListener('click', (e) => {
        if (e.target === startHandle || e.target === endHandle || e.target.classList.contains('time-label')) return;
        const mouseX = getMousePositionInScrubber(e);
        const clickTime = (mouseX / scrubberWidth) * videoDuration;
        videoPlayer.currentTime = clickTime;
    });

    playSegmentBtn.addEventListener('click', () => {
        videoPlayer.removeEventListener('timeupdate', segmentPlayHandler);
        videoPlayer.currentTime = editStartTime;
        videoPlayer.play();
        videoPlayer.addEventListener('timeupdate', segmentPlayHandler);
    });

    // --- 헬퍼 함수 ---
    function handleFiles(files) {
        const placeholder = videoListContainer.querySelector('.video-list-placeholder');
        if (placeholder) placeholder.style.display = 'none';
        for (const file of files) {
            if (file.type.startsWith('video/')) addVideoToList(file);
        }
    }

    function addVideoToList(file) {
        const videoURL = URL.createObjectURL(file);

        const thumbnailDiv = document.createElement('div');
        thumbnailDiv.className = 'video-thumbnail';
        thumbnailDiv.innerHTML = `
            <img src="" alt="Thumbnail" style="width: 160px; height: 90px; object-fit: cover; background-color: #222;">
            <div class="video-thumbnail-name">${file.name}</div>
        `;
        const thumbnailImg = thumbnailDiv.querySelector('img');

        const tempVideo = document.createElement('video');
        tempVideo.muted = true;
        tempVideo.style.display = 'none';

        const onSeeked = () => {
            setTimeout(() => {
                const canvas = document.createElement('canvas');
                canvas.width = 160;
                canvas.height = 90;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(tempVideo, 0, 0, canvas.width, canvas.height);
                thumbnailImg.src = canvas.toDataURL('image/jpeg');
                if (document.body.contains(tempVideo)) document.body.removeChild(tempVideo);
            }, 100);
        };

        const onLoadedData = () => {
            const seekTime = (tempVideo.duration > 1) ? 1.0 : (tempVideo.duration / 2);
            tempVideo.currentTime = seekTime;
        };

        const onError = () => {
            console.error("썸네일 생성 실패:", file.name);
            if (document.body.contains(tempVideo)) document.body.removeChild(tempVideo);
        };

        tempVideo.addEventListener('loadeddata', onLoadedData, { once: true });
        tempVideo.addEventListener('seeked', onSeeked, { once: true });
        tempVideo.addEventListener('error', onError, { once: true });

        document.body.appendChild(tempVideo);
        tempVideo.src = videoURL;

        thumbnailDiv.addEventListener('click', () => {
            playerPlaceholder.style.display = 'none';
            videoPlayer.style.display = 'block';
            videoPlayer.src = videoURL;
            analysisResult.innerText = "Select a video segment to analyze for deception.";
            currentSelectedFile = file;
            
            // [유지] 파싱 관련 UI 리셋
            clipListContainer.innerHTML = "<p class='clip-list-placeholder'>Run 'Segment Video' to generate sentence segments.</p>";
            allSegments = [];
            if (segmentListTitle) {
                segmentListTitle.innerText = "Sentence Segments";
            }
        });

        videoListContainer.appendChild(thumbnailDiv);
    }

    function generateScrubberThumbnails() {
        timelineScrubber.innerHTML = '';
        timelineScrubber.appendChild(selectedRange);
        timelineScrubber.appendChild(startHandle);
        timelineScrubber.appendChild(endHandle);
        timelineScrubber.appendChild(currentTimeMarker);

        const segmentDuration = 2;
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
                if (document.body.contains(thumbVideo)) document.body.removeChild(thumbVideo);
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

    function getMousePositionInScrubber(event) {
        const rect = timelineScrubber.getBoundingClientRect();
        let x = event.clientX - rect.left;
        return Math.max(0, Math.min(x, scrubberWidth));
    }

    const segmentPlayHandler = () => {
        if (videoPlayer.currentTime >= editEndTime) {
            videoPlayer.pause();
            videoPlayer.removeEventListener('timeupdate', segmentPlayHandler);
        }
    };

    function updateScrubberUI() {
        const currentScrubberWidth = timelineScrubber.offsetWidth;
        if (!videoDuration || currentScrubberWidth === 0) return;

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

    function formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) seconds = 0;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        const ms = Math.floor((seconds - Math.floor(seconds)) * 100);
        return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
    }

    new ResizeObserver(() => {
        updateScrubberUI();
    }).observe(timelineScrubber);

    // [팝업 추가] 모달 닫기 이벤트 리스너
    modalCloseBtn.addEventListener('click', () => {
        resultModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target == resultModal) {
            resultModal.style.display = 'none';
        }
    });

}); // DOMContentLoaded 종료
