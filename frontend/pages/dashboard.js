let dashboardLogs = [
    { name: "LLM_Architecture_DeepDive.mp4", detail: "Transcription Complete • 48:22 min", time: "2 mins ago", status: "complete" },
    { name: "TensorFlow_Optimization_V2.mov", detail: "Ready for Multimodal Chat", time: "1 hour ago", status: "complete" }
];

function renderDashboard() {
    const isPro = window.currentUser && window.currentUser.role === 'Pro';
    return `
    <main class="flex-grow overflow-y-auto scrollbar-hide p-container-padding bg-background h-full">
        <div class="max-w-5xl mx-auto space-y-gutter">
            <!-- Primary Application Header -->
            <section class="mb-6">
                <h2 class="font-headline-lg text-headline-lg text-on-surface mb-2">Video Ingestion Dashboard</h2>
                <p class="font-body-md text-body-md text-on-surface-variant max-w-2xl">
                    Seamlessly process high-fidelity video streams. Our AI-driven multimodal engine provides instantaneous transcription, scene analysis, and semantic indexing for technical content.
                </p>
            </section>

            <!-- Primary Dashboard Grid Layout -->
            <div class="grid grid-cols-12 gap-gutter">
                <!-- Ingestion Source Entry Component -->
                <div class="col-span-12 lg:col-span-8 space-y-gutter">
                    <div class="glass-panel p-stack-lg rounded-xl shadow-2xl relative overflow-hidden group">
                        <div class="absolute top-0 right-0 p-4 opacity-10">
                            <span class="material-symbols-outlined text-[120px]">movie_edit</span>
                        </div>
                        <h3 class="font-label-mono text-label-mono text-primary uppercase tracking-widest mb-4">Video Source URL</h3>
                        <div class="flex flex-col gap-4">
                            <div class="relative flex items-center bg-surface-container-lowest border border-outline-variant rounded-lg">
                                <span class="material-symbols-outlined absolute left-4 text-outline">link</span>
                                <input id="ingest-url-input" class="w-full bg-transparent border-none focus:ring-0 py-4 pl-12 pr-4 font-body-md text-body-md text-on-surface outline-none" placeholder="https://youtube.com/watch?v=... or Google Drive direct link" type="text">
                            </div>
                            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                                <span class="font-body-sm text-[12px] text-on-surface-variant italic">Supports YouTube, Vimeo, and Google Drive Direct Links</span>
                                <button onclick="triggerUrlIngestion()" class="bg-primary-container text-on-primary-container px-8 py-3 rounded-lg font-bold text-body-md hover:bg-primary transition-all flex items-center gap-2">
                                    <span class="material-symbols-outlined">play_circle</span>
                                    PROCESS URL
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Direct Upload Ingestion Component -->
                    <div class="glass-panel p-stack-lg rounded-xl shadow-2xl relative overflow-hidden">
                        <h3 class="font-label-mono text-label-mono text-primary uppercase tracking-widest mb-4">Direct File Upload</h3>
                        <div class="border-2 border-dashed border-outline-variant/50 hover:border-primary/50 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer relative group transition-all" onclick="document.getElementById('file-upload-input').click()">
                            <input type="file" id="file-upload-input" class="hidden" accept="video/*" onchange="handleFileUpload(this)">
                            <div class="w-12 h-12 rounded-full bg-primary-container/10 flex items-center justify-center text-primary mb-4 group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-3xl">upload_file</span>
                            </div>
                            <p class="font-headline-md text-body-md text-on-surface font-semibold mb-1" id="upload-status-text">Drag & drop your video file here</p>
                            <p class="font-body-sm text-[12px] text-on-surface-variant">Supports MP4, MOV, MKV up to 100MB</p>
                        </div>
                    </div>

                    <!-- Historical Ingestion Logs -->
                    <div class="glass-panel p-stack-lg rounded-xl">
                        <div class="flex justify-between items-center mb-6">
                            <h3 class="font-label-mono text-label-mono text-primary uppercase tracking-widest">Recent Task Logs</h3>
                            <button onclick="refreshVideos()" class="text-primary font-body-sm text-body-sm hover:underline flex items-center gap-1">
                                <span class="material-symbols-outlined text-[16px]">refresh</span> Refresh
                            </button>
                        </div>
                        <div class="space-y-3" id="task-logs-container">
                            <!-- Hydrated dynamically with task history -->
                        </div>
                    </div>
                </div>

                <!-- System Telemetry and Statistics Panel -->
                <div class="col-span-12 lg:col-span-4 space-y-gutter">
                    <!-- Engine Status Overview -->
                    <div class="glass-panel p-stack-lg rounded-xl space-y-6">
                        <h3 class="font-label-mono text-label-mono text-primary uppercase tracking-widest">Engine Status</h3>
                        
                        <div class="space-y-4">
                            <div class="flex justify-between items-center pb-3 border-b border-outline-variant/20">
                                <span class="text-on-surface-variant text-[14px]">Qdrant Database</span>
                                <span class="flex items-center gap-1.5 text-green-400 text-xs font-semibold">
                                    <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span> Connected
                                </span>
                            </div>
                            <div class="flex justify-between items-center pb-3 border-b border-outline-variant/20">
                                <span class="text-on-surface-variant text-[14px]">Gemini Vision API</span>
                                <span class="flex items-center gap-1.5 text-green-400 text-xs font-semibold">
                                    <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span> Operational
                                </span>
                            </div>
                            <div class="flex justify-between items-center pb-3 border-b border-outline-variant/20">
                                <span class="text-on-surface-variant text-[14px]">Transcription Engine</span>
                                <span class="flex items-center gap-1.5 text-green-400 text-xs font-semibold">
                                    <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span> Active
                                </span>
                            </div>
                        </div>

                        <!-- Resource Utilization Metrics -->
                        <div class="pt-2">
                            <p class="text-on-surface-variant font-label-mono text-[10px] uppercase mb-3">Resource Load</p>
                            <div class="space-y-3">
                                <div>
                                    <div class="flex justify-between text-xs text-on-surface-variant mb-1">
                                        <span>GPU Allocation</span>
                                        <span>38%</span>
                                    </div>
                                    <div class="h-1 bg-surface-container rounded-full overflow-hidden">
                                        <div class="bg-primary h-full w-[38%]"></div>
                                    </div>
                                </div>
                                <div>
                                    <div class="flex justify-between text-xs text-on-surface-variant mb-1">
                                        <span>Indexed Vectors</span>
                                        <span id="indexed-vectors-count">1,248</span>
                                    </div>
                                    <div class="h-1 bg-surface-container rounded-full overflow-hidden">
                                        <div class="bg-secondary h-full w-[65%]"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Subscription Upgrade Promotion -->
                    <div class="p-6 bg-gradient-to-br from-primary-container/20 to-surface-container rounded-xl border border-primary/20 relative overflow-hidden">
                        <div class="absolute -right-4 -bottom-4 opacity-10">
                            <span class="material-symbols-outlined text-[100px]">bolt</span>
                        </div>
                        <h4 class="font-label-mono text-[10px] text-primary uppercase tracking-widest mb-1">Upgrade Tier</h4>
                        <p class="font-headline-md text-body-md font-bold text-on-surface mb-2">Need unlimited processing?</p>
                        <p class="text-[12px] text-on-surface-variant mb-4 leading-relaxed">
                            Activate the Pro Insight plan for infinite video processing lengths, direct API key access, and concurrent ingestion pipelines.
                        </p>
                        <button onclick="navigate('pricing')" class="bg-primary text-on-primary font-semibold text-xs px-4 py-2 rounded-lg hover:bg-white hover:text-primary transition-all">
                            View Pricing Plans
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </main>
    `;
}

function initDashboard() {
    refreshVideos();
}

async function refreshVideos() {
    const container = document.getElementById('task-logs-container');
    if (!container) return;

    container.innerHTML = `
        <div class="flex items-center justify-center py-6 text-on-surface-variant">
            <span class="material-symbols-outlined animate-spin mr-2">sync</span> Loading task history...
        </div>
    `;

    try {
        const response = await fetch(`${window.API_BASE_URL}/api/videos`);
        if (!response.ok) throw new Error("Could not load video library");
        
        const videos = await response.json();
        
        // Merge fetched ingestion tasks with static placeholders
        const allLogs = [];
        
        // Process dynamic ingestion tasks
        videos.forEach((vid, index) => {
            allLogs.push({
                name: vid.name || `Video_${vid.id}`,
                detail: `ID: ${vid.id} • ${vid.chunks_count || 0} semantic vectors`,
                time: "Ready to query",
                status: "complete",
                id: vid.id
            });
        });

        // Append static placeholders for UI stability
        dashboardLogs.forEach(log => {
            allLogs.push(log);
        });

        // Update global workspace counter
        const counter = document.getElementById('projects-counter');
        if (counter) {
            counter.textContent = `${videos.length}/5 Projects`;
        }

        // Render task components
        container.innerHTML = allLogs.map((log, index) => `
            <div class="flex items-center justify-between p-4 bg-surface-container-low border border-outline-variant/30 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer" onclick="openChatFromLog('${log.id || ''}')">
                <div class="flex items-center gap-4 min-w-0 flex-1">
                    <div class="w-8 h-8 shrink-0 rounded-full bg-green-500/20 flex items-center justify-center text-green-400">
                        <span class="material-symbols-outlined text-[18px]" style="font-variation-settings: 'FILL' 1;">check_circle</span>
                    </div>
                    <div class="min-w-0 flex-1">
                        <p class="font-body-md text-body-md text-on-surface font-medium truncate">${log.name}</p>
                        <p class="font-code-sm text-[12px] text-on-surface-variant truncate">${log.detail}</p>
                    </div>
                </div>
                <div class="text-right flex items-center gap-4">
                    <span class="font-code-sm text-xs text-outline">${log.time}</span>
                    ${log.id ? `
                        <button onclick="event.stopPropagation(); openChatFromLog('${log.id}')" class="text-xs bg-primary-container/20 text-primary-fixed-dim border border-primary/30 px-3 py-1 rounded hover:bg-primary-container transition-all">
                            Chat
                        </button>
                        <button onclick="event.stopPropagation(); deleteVideo('${log.id}')" class="text-xs bg-error-container/20 text-error border border-error/30 px-3 py-1 rounded hover:bg-error hover:text-white transition-all">
                            Delete
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error("Error refreshing task logs:", err);
        container.innerHTML = `
            <div class="p-4 bg-error-container/20 border border-error/20 text-error rounded-lg text-center text-xs">
                Failed to fetch task history. Check API server connectivity.
            </div>
        `;
    }
}

function openChatFromLog(videoId) {
    if (!videoId) return;
    window.activeVideoId = videoId;
    navigate('chat');
}

async function deleteVideo(videoId) {
    if (!confirm("Are you sure you want to delete this video and all its AI data? This cannot be undone.")) return;
    
    try {
        const response = await fetch(`${window.API_BASE_URL}/api/videos/${videoId}`, { method: 'DELETE' });
        if (response.ok) {
            if (window.activeVideoId === videoId) {
                window.activeVideoId = null;
            }
            refreshVideos();
        } else {
            const data = await response.json();
            alert("Failed to delete: " + (data.detail || "Unknown error"));
        }
    } catch (err) {
        console.error("Delete error:", err);
        alert("Failed to connect to the server to delete the video.");
    }
}

async function triggerUrlIngestion() {
    const urlInput = document.getElementById('ingest-url-input');
    const url = urlInput.value.trim();
    if (!url) return;

    urlInput.disabled = true;
    
    // Render optimistic UI state for pending task
    const container = document.getElementById('task-logs-container');
    const tempId = 'temp_' + Date.now();
    const tempLogHtml = `
        <div id="${tempId}" class="flex items-center justify-between p-4 bg-surface-container-low border border-outline-variant/30 rounded-lg">
            <div class="flex items-center gap-4">
                <div class="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                    <span class="material-symbols-outlined text-[18px] animate-spin">sync</span>
                </div>
                <div>
                    <p class="font-body-md text-body-md text-on-surface font-medium truncate max-w-md">${url}</p>
                    <p class="font-code-sm text-[12px] text-primary" id="${tempId}_status">Initializing Ingestion Pipeline...</p>
                </div>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('afterbegin', tempLogHtml);

    try {
        const response = await fetch(`${window.API_BASE_URL}/api/ingest/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_url: url })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "URL ingestion trigger failed");

        const callId = data.call_id;
        pollIngestionStatus(callId, tempId, data.video_id);
        
        urlInput.value = "";
    } catch (err) {
        console.error("URL Ingestion Error:", err);
        const statusText = document.getElementById(`${tempId}_status`);
        if (statusText) {
            statusText.textContent = err.message || "Failed to trigger ingestion";
            statusText.className = "font-code-sm text-[12px] text-error";
        }
    } finally {
        urlInput.disabled = false;
    }
}

async function handleFileUpload(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const statusText = document.getElementById('upload-status-text');
    
    statusText.textContent = `Uploading: ${file.name} (0%)`;
    
    const formData = new FormData();
    formData.append('file', file);

    const container = document.getElementById('task-logs-container');
    const tempId = 'temp_' + Date.now();
    const tempLogHtml = `
        <div id="${tempId}" class="flex items-center justify-between p-4 bg-surface-container-low border border-outline-variant/30 rounded-lg">
            <div class="flex items-center gap-4">
                <div class="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                    <span class="material-symbols-outlined text-[18px] animate-spin">sync</span>
                </div>
                <div>
                    <p class="font-body-md text-body-md text-on-surface font-medium truncate max-w-md">${file.name}</p>
                    <p class="font-code-sm text-[12px] text-primary" id="${tempId}_status">Uploading file to server...</p>
                </div>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('afterbegin', tempLogHtml);

    try {
        const response = await fetch(`${window.API_BASE_URL}/api/ingest/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Upload ingestion failed");

        const callId = data.call_id;
        pollIngestionStatus(callId, tempId, data.video_id);
        
        statusText.textContent = "Drag & drop your video file here";
    } catch (err) {
        console.error("Upload Ingestion Error:", err);
        const statusTextLog = document.getElementById(`${tempId}_status`);
        if (statusTextLog) {
            statusTextLog.textContent = err.message || "Failed to upload video";
            statusTextLog.className = "font-code-sm text-[12px] text-error";
        }
        statusText.textContent = "Upload failed. Try again.";
    }
}

function pollIngestionStatus(callId, tempId, videoId) {
    const statusText = document.getElementById(`${tempId}_status`);
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`${window.API_BASE_URL}/api/ingest/status/${callId}`);
            if (!resp.ok) return;
            
            const data = await resp.json();
            
            if (data.status === 'complete') {
                clearInterval(interval);
                if (statusText) {
                    statusText.textContent = "Processing Completed successfully.";
                    statusText.className = "font-code-sm text-[12px] text-green-400";
                }
                setTimeout(() => {
                    refreshVideos();
                }, 1000);
            } else if (data.status === 'error' || data.status === 'expired' || data.status === 'timeout') {
                clearInterval(interval);
                if (statusText) {
                    statusText.textContent = `Pipeline failure: ${data.detail || 'unknown error'}`;
                    statusText.className = "font-code-sm text-[12px] text-error";
                }
            } else {
                if (statusText) {
                    statusText.textContent = `Processing on Modal... Running (${data.status})`;
                }
            }
        } catch (err) {
            console.error("Status polling error:", err);
        }
    }, 4000);
}
