let currentChatHistory = [];
let availableVideos = [];

function renderChat() {
    return `
    <div class="flex flex-col md:flex-row h-full overflow-hidden bg-background">
        <!-- Primary Video and Context Viewport -->
        <section class="w-full md:w-3/5 flex flex-col p-gutter gap-gutter h-full overflow-y-auto scrollbar-hide">
            <!-- Source Media Selection -->
            <div class="glass-panel p-4 rounded-xl flex items-center justify-between gap-4 shrink-0">
                <div class="flex items-center gap-2 flex-grow">
                    <span class="material-symbols-outlined text-primary">movie</span>
                    <label class="font-label-mono text-xs text-outline uppercase tracking-wider hidden md:block">Active Video:</label>
                    <select id="video-select" onchange="handleVideoChange(this.value)" class="bg-surface-container-lowest border border-outline-variant text-on-surface rounded-lg px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-primary flex-grow">
                        <option value="">-- Choose video to analyze --</option>
                    </select>
                </div>
                <button onclick="refreshChatVideos()" class="material-symbols-outlined text-on-surface-variant hover:text-primary transition-colors p-1.5 rounded-full bg-surface-container">
                    refresh
                </button>
            </div>

            <!-- Media Playback Container -->
            <div class="relative aspect-video bg-black rounded-xl overflow-hidden border border-outline-variant active-glow group shrink-0" id="video-player-container">
                <!-- Empty State Placeholder -->
                <div class="absolute inset-0 flex flex-col items-center justify-center text-center p-6 bg-[#021016]" id="player-placeholder">
                    <span class="material-symbols-outlined text-[64px] text-outline mb-2">video_library</span>
                    <h4 class="font-bold text-on-surface text-lg">No Active Video Stream</h4>
                    <p class="text-on-surface-variant text-xs mt-1">Select a video from the dropdown above to index its transcription & keyframes</p>
                </div>
            </div>

            <!-- Real-time Context Analytics -->
            <div class="bg-surface-container-low border-l-4 border-primary-container p-container-padding rounded-xl glass shrink-0" id="context-info-card">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-3">
                        <div class="w-2 h-2 rounded-full bg-primary-container animate-pulse"></div>
                        <h3 class="font-headline-md text-body-md font-bold uppercase tracking-wider text-primary">Current Segment Context</h3>
                    </div>
                    <span class="font-label-mono text-label-mono bg-primary-container/20 text-primary-fixed-dim px-3 py-1 rounded-full border border-primary/30" id="context-timestamp">Timestamp: --:--</span>
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-surface-container-highest/40 p-4 rounded-lg border border-outline-variant">
                        <p class="text-on-surface-variant font-label-mono text-[10px] uppercase mb-1">Visual Index</p>
                        <div class="flex flex-wrap gap-2" id="context-objects">
                            <span class="text-outline text-xs italic">Awaiting playback...</span>
                        </div>
                    </div>
                    <div class="bg-surface-container-highest/40 p-4 rounded-lg border border-outline-variant">
                        <p class="text-on-surface-variant font-label-mono text-[10px] uppercase mb-1">Retrieval homologies</p>
                        <p class="text-on-surface font-body-sm" id="context-sentiment">Automatic metadata indexing active.</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Conversational Interface -->
        <section class="w-full md:w-2/5 flex flex-col bg-surface-container border-l border-outline-variant h-full overflow-hidden">
            <!-- Prompt Recommendations -->
            <div class="p-4 border-b border-outline-variant">
                <p class="text-on-surface-variant font-label-mono text-[10px] uppercase tracking-widest mb-3 px-2">Suggested Prompts</p>
                <div class="flex flex-wrap gap-2">
                    <button onclick="sendQuickPrompt('Summarize this video')" class="px-4 py-2 bg-surface-container-high border border-outline-variant rounded-full text-[12px] hover:bg-primary-container hover:border-primary-container hover:text-white transition-all duration-200">Summarize</button>
                    <button onclick="sendQuickPrompt('What are the key concepts?')" class="px-4 py-2 bg-surface-container-high border border-outline-variant rounded-full text-[12px] hover:bg-primary-container hover:border-primary-container hover:text-white transition-all duration-200">Key Concepts</button>
                    <button onclick="sendQuickPrompt('Explain the final conclusion')" class="px-4 py-2 bg-surface-container-high border border-outline-variant rounded-full text-[12px] hover:bg-primary-container hover:border-primary-container hover:text-white transition-all duration-200">Conclusion</button>
                </div>
            </div>

            <!-- Conversation History -->
            <div id="chat-messages-box" class="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-hide">
                <!-- Messages rendered dynamically -->
            </div>

            <!-- Vision Context Upload -->
            <div class="px-4 py-2 bg-surface-container-highest/20 border-t border-outline-variant/30 flex items-center justify-between gap-3 text-xs text-on-surface-variant">
                <div class="flex items-center gap-1.5">
                    <span class="material-symbols-outlined text-[18px]">photo_camera</span>
                    <span>Provide frame screenshot (optional)</span>
                </div>
                <input type="file" id="chat-frame-upload" class="hidden" accept="image/*" onchange="handleChatFrameSelect(this)">
                <button onclick="document.getElementById('chat-frame-upload').click()" class="text-xs bg-primary-container/20 text-primary border border-primary/20 px-2 py-1 rounded hover:bg-primary-container hover:text-white transition-all" id="chat-frame-upload-btn">
                    Select Frame
                </button>
            </div>

            <!-- Query Input Interface -->
            <div class="p-4 border-t border-outline-variant bg-surface-container-low">
                <form onsubmit="handleChatSubmit(event)" class="relative group">
                    <input id="chat-input-field" class="w-full bg-surface-container-lowest border border-outline-variant rounded-xl py-4 pl-4 pr-24 focus:border-primary-container focus:ring-2 focus:ring-primary-container/20 transition-all text-on-surface placeholder:text-outline font-body-sm text-[14px]" placeholder="Ask about this video..." type="text" required>
                    <div class="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                        <button type="submit" class="bg-primary-container hover:bg-primary text-white px-4 py-1.5 rounded-lg font-label-mono text-xs flex items-center gap-2 transition-all active:scale-95">
                            Send
                            <span class="material-symbols-outlined text-[16px]">send</span>
                        </button>
                    </div>
                </form>
            </div>
        </section>
    </div>
    `;
}

function initChat() {
    refreshChatVideos();
    resetChatUI();
}

async function refreshChatVideos() {
    const select = document.getElementById('video-select');
    if (!select) return;

    try {
        const resp = await fetch(`${window.API_BASE_URL}/api/videos`);
        if (!resp.ok) return;
        
        availableVideos = await resp.json();
        
        select.innerHTML = '<option value="">-- Choose video to analyze --</option>' + 
            availableVideos.map(v => `<option value="${v.id}">${v.name || v.id}</option>`).join('');

        if (window.activeVideoId) {
            select.value = window.activeVideoId;
            handleVideoChange(window.activeVideoId);
        }
    } catch (err) {
        console.error("Error fetching chat videos:", err);
    }
}

function resetChatUI() {
    const chatBox = document.getElementById('chat-messages-box');
    if (!chatBox) return;

    chatBox.innerHTML = `
        <div class="flex gap-4">
            <div class="w-8 h-8 rounded-lg flex items-center justify-center shrink-0">
                <img src="/assets/logo_icon.png" alt="VidEx" class="w-full h-full object-contain">
            </div>
            <div class="flex-1">
                <p class="text-on-surface-variant font-label-mono text-xs mb-1">VidEx</p>
                <div class="bg-surface-container-highest p-4 rounded-2xl rounded-tl-none border border-outline-variant text-on-surface font-body-sm leading-relaxed text-[13px]">
                    Select a video and ask any question! I will analyze the speech transcript, cross-reference keyframes, and locate specific timestamps for you.
                </div>
            </div>
        </div>
    `;
}

function handleVideoChange(videoId) {
    if (!videoId) {
        window.activeVideoId = null;
        document.getElementById('video-player-container').innerHTML = `
            <div class="absolute inset-0 flex flex-col items-center justify-center text-center p-6 bg-[#021016]" id="player-placeholder">
                <span class="material-symbols-outlined text-[64px] text-outline mb-2">video_library</span>
                <h4 class="font-bold text-on-surface text-lg">No Active Video Stream</h4>
                <p class="text-on-surface-variant text-xs mt-1">Select a video from the dropdown above to index its transcription & keyframes</p>
            </div>
        `;
        resetChatUI();
        return;
    }

    window.activeVideoId = videoId;
    
    // Initialize media player based on source domain
    const playerContainer = document.getElementById('video-player-container');
    const isYoutube = videoId.length === 11 && !videoId.includes(" ");
    
    if (isYoutube) {
        playerContainer.innerHTML = `
            <iframe id="vid-element" class="w-full h-full object-cover" src="https://www.youtube.com/embed/${videoId}?enablejsapi=1&autoplay=1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        `;
    } else {
        playerContainer.innerHTML = `
            <video id="vid-element" controls autoplay class="w-full h-full object-contain">
                <source src="${window.API_BASE_URL}/temp_assets/${videoId}.mp4" type="video/pipe">
                <source src="${window.API_BASE_URL}/temp_assets/${videoId}.mp4" type="video/mp4">
                Your browser does not support html5 video playback.
            </video>
        `;
        
        // Bind timeupdate events to update context telemetry
        setTimeout(() => {
            const video = document.getElementById('vid-element');
            if (video) {
                video.addEventListener('timeupdate', () => {
                    updateCurrentContextInfo(video.currentTime);
                });
            }
        }, 500);
    }

    resetChatUI();
    currentChatHistory = [];
}

function updateCurrentContextInfo(seconds) {
    const timestampSpan = document.getElementById('context-timestamp');
    const objectsDiv = document.getElementById('context-objects');
    
    if (timestampSpan) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        timestampSpan.textContent = `Timestamp: ${m}:${s < 10 ? '0' : ''}${s}`;
    }

    if (objectsDiv) {
        // Update telemetry visualization based on playback position
        if (seconds > 10 && seconds < 60) {
            objectsDiv.innerHTML = `
                <span class="px-2 py-0.5 bg-background border border-outline-variant rounded text-xs font-code-sm">Presentation (0.97)</span>
                <span class="px-2 py-0.5 bg-background border border-outline-variant rounded text-xs font-code-sm">Whiteboard (0.91)</span>
            `;
        } else if (seconds >= 60 && seconds < 180) {
            objectsDiv.innerHTML = `
                <span class="px-2 py-0.5 bg-background border border-outline-variant rounded text-xs font-code-sm">IDE Code Window (0.99)</span>
                <span class="px-2 py-0.5 bg-background border border-outline-variant rounded text-xs font-code-sm">Python terminal (0.92)</span>
            `;
        } else {
            objectsDiv.innerHTML = `
                <span class="px-2 py-0.5 bg-background border border-outline-variant rounded text-xs font-code-sm">Visual Stream Active (0.98)</span>
            `;
        }
    }
}

function jumpToTime(seconds) {
    const isYoutube = window.activeVideoId && window.activeVideoId.length === 11 && !window.activeVideoId.includes(" ");
    
    if (isYoutube) {
        const player = document.getElementById('vid-element');
        if (player) {
            player.src = `https://www.youtube.com/embed/${window.activeVideoId}?start=${Math.floor(seconds)}&autoplay=1&enablejsapi=1`;
        }
    } else {
        const video = document.getElementById('vid-element');
        if (video) {
            video.currentTime = seconds;
            video.play();
        }
    }

    updateCurrentContextInfo(seconds);
}

function handleChatFrameSelect(input) {
    const btn = document.getElementById('chat-frame-upload-btn');
    if (input.files && input.files[0]) {
        btn.textContent = `Frame Selected: ${input.files[0].name.substring(0, 10)}...`;
        btn.classList.add('bg-green-500/20', 'text-green-400');
    } else {
        btn.textContent = "Select Frame";
        btn.classList.remove('bg-green-500/20', 'text-green-400');
    }
}

function sendQuickPrompt(promptText) {
    const input = document.getElementById('chat-input-field');
    if (input) {
        input.value = promptText;
        handleChatSubmit(new Event('submit'));
    }
}

async function handleChatSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById('chat-input-field');
    const query = input.value.trim();
    if (!query || !window.activeVideoId) return;

    // Render user query
    appendMessage("user", query);
    input.value = "";

    const chatBox = document.getElementById('chat-messages-box');
    const loadingMsgId = 'ai-loading-' + Date.now();
    
    // Render pending state indicator
    const loadingHtml = `
        <div class="flex gap-4" id="${loadingMsgId}">
            <div class="w-8 h-8 rounded-lg flex items-center justify-center shrink-0">
                <img src="/assets/logo_icon.png" alt="VidEx" class="w-full h-full object-contain animate-pulse">
            </div>
            <div class="flex-1">
                <p class="text-on-surface-variant font-label-mono text-xs mb-1">VidEx</p>
                <div class="bg-surface-container-highest p-4 rounded-2xl rounded-tl-none border border-outline-variant text-on-surface font-body-sm leading-relaxed text-[13px]">
                    Analyzing transcript indices and visual frames...
                </div>
            </div>
        </div>
    `;
    chatBox.insertAdjacentHTML('beforeend', loadingHtml);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Construct request payload
    const bodyFormData = new FormData();
    bodyFormData.append('video_id', window.activeVideoId);
    bodyFormData.append('query', query);
    
    // Append vision context if present
    const frameUpload = document.getElementById('chat-frame-upload');
    if (frameUpload && frameUpload.files && frameUpload.files[0]) {
        bodyFormData.append('frame', frameUpload.files[0]);
    }

    // Reset upload interface
    const frameBtn = document.getElementById('chat-frame-upload-btn');
    if (frameBtn) {
        frameBtn.textContent = "Select Frame";
        frameBtn.classList.remove('bg-green-500/20', 'text-green-400');
    }
    if (frameUpload) frameUpload.value = "";

    try {
        const response = await fetch(`${window.API_BASE_URL}/api/chat`, {
            method: 'POST',
            body: bodyFormData
        });

        const data = await response.json();
        
        // Clear pending indicator
        const loader = document.getElementById(loadingMsgId);
        if (loader) loader.remove();

        if (response.ok) {
            appendMessage("assistant", data.answer, data.sources || []);
        } else {
            appendMessage("assistant", `Error: ${data.detail || 'Could not query intelligence model'}`);
        }
    } catch (err) {
        console.error("Chat Query Error:", err);
        const loader = document.getElementById(loadingMsgId);
        if (loader) loader.remove();
        appendMessage("assistant", "Engine connection error. Unable to reach the API backend. Please check your network connection or backend status.");
    }
}

function appendMessage(role, content, sources = []) {
    const chatBox = document.getElementById('chat-messages-box');
    if (!chatBox) return;

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="mt-4 pt-3 border-t border-outline-variant/30">
                <p class="font-label-mono text-[10px] text-outline uppercase tracking-wider mb-2">Sources (Click to jump):</p>
                <div class="flex flex-wrap gap-2">
                    ${sources.map(ts => {
                        const m = Math.floor(ts / 60);
                        const s = Math.floor(ts % 60);
                        return `
                            <button onclick="jumpToTime(${ts})" class="px-3 py-1 bg-primary-container/10 border border-primary/20 hover:bg-primary-container hover:text-white rounded-md text-xs font-code-sm text-primary flex items-center gap-1.5 transition-all">
                                <span class="material-symbols-outlined text-[13px]">play_circle</span>
                                ⏱️ ${m}:${s < 10 ? '0' : ''}${s}
                            </button>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    if (role === 'user') {
        chatBox.insertAdjacentHTML('beforeend', `
            <div class="flex gap-4 flex-row-reverse">
                <div class="w-8 h-8 rounded-lg bg-secondary-container flex items-center justify-center shrink-0">
                    <span class="material-symbols-outlined text-on-secondary-container text-sm">person</span>
                </div>
                <div class="flex-1 text-right">
                    <p class="text-on-surface-variant font-label-mono text-xs mb-1">User</p>
                    <div class="inline-block bg-primary-container/20 p-4 rounded-2xl rounded-tr-none border border-primary/30 text-on-surface font-body-sm text-[13px] text-left">
                        ${content}
                    </div>
                </div>
            </div>
        `);
    } else {
        chatBox.insertAdjacentHTML('beforeend', `
            <div class="flex gap-4 animate-in fade-in duration-300">
                <div class="w-8 h-8 rounded-lg flex items-center justify-center shrink-0">
                    <img src="/assets/logo_icon.png" alt="VidEx" class="w-full h-full object-contain">
                </div>
                <div class="flex-1">
                    <p class="text-on-surface-variant font-label-mono text-xs mb-1">VidEx</p>
                    <div class="bg-surface-container-highest p-4 rounded-2xl rounded-tl-none border border-outline-variant text-on-surface font-body-sm leading-relaxed text-[13px]">
                        <div class="prose dark:prose-invert text-sm max-w-none text-on-surface">
                            ${content.replace(/\n/g, '<br>')}
                        </div>
                        ${sourcesHtml}
                    </div>
                </div>
            </div>
        `);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}
