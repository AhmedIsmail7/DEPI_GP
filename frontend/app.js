/* ==================== Application Router & State Manager ==================== */

const NAV_ITEMS = [
    { id: 'dashboard', icon: 'upload_file', label: 'Ingestion Dashboard' },
    { id: 'chat',      icon: 'forum',        label: 'Multimodal Chat' },
    { id: 'pricing',   icon: 'payments',     label: 'Pricing Plans' }
];

const AUTH_PAGES = ['login', 'signup'];
let currentPage = '';

// Global session state
window.API_BASE_URL = "https://videx-production-1994.up.railway.app";
window.currentUser = JSON.parse(localStorage.getItem('currentUser')) || null;
window.activeVideoId = null;

function navigate(page) {
    // Enforce authentication requirements for protected routes
    if (!window.currentUser && !AUTH_PAGES.includes(page)) {
        page = 'login';
    }
    
    // Redirect authenticated users away from authentication views
    if (window.currentUser && AUTH_PAGES.includes(page)) {
        page = 'dashboard';
    }

    currentPage = page;
    window.location.hash = page;
    
    renderPage();
}

function renderPage() {
    const authContainer = document.getElementById('auth-container');
    const appShell = document.getElementById('app-shell');
    const pageContent = document.getElementById('page-content');

    // Render full-viewport authentication layouts
    if (AUTH_PAGES.includes(currentPage)) {
        appShell.classList.add('hidden');
        authContainer.classList.remove('hidden');
        authContainer.className = "min-h-screen w-full flex items-center justify-center relative";

        if (currentPage === 'login') {
            authContainer.innerHTML = renderLogin();
        } else if (currentPage === 'signup') {
            authContainer.innerHTML = renderSignup();
        }
    } else {
        // Render main application shell
        authContainer.classList.add('hidden');
        appShell.classList.remove('hidden');
        appShell.classList.add('flex');
        
        renderSidebarNav();
        initSession();

        if (currentPage === 'dashboard') {
            pageContent.innerHTML = renderDashboard();
            initDashboard();
        } else if (currentPage === 'chat') {
            pageContent.innerHTML = renderChat();
            initChat();
        } else if (currentPage === 'pricing') {
            pageContent.innerHTML = renderPricing();
        }
    }
}

function renderSidebarNav() {
    const nav = document.getElementById('sidebar-nav');
    if (!nav) return;

    nav.innerHTML = NAV_ITEMS.map(item => {
        const isActive = currentPage === item.id;
        const activeClass = isActive 
            ? 'bg-primary-container text-on-primary-container font-bold scale-[0.98]' 
            : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest';
        
        return `
            <a onclick="navigate('${item.id}'); return false;" class="flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 cursor-pointer ${activeClass}">
                <span class="material-symbols-outlined">${item.icon}</span>
                <span class="font-body-sm text-body-sm">${item.label}</span>
            </a>
        `;
    }).join('');
}

function initSession() {
    if (!window.currentUser) return;
    
    // Hydrate user interface with session data
    const userBadge = document.getElementById('user-role-badge');
    const projectCount = document.getElementById('projects-counter');
    
    if (userBadge) {
        userBadge.textContent = window.currentUser.role || 'Free';
    }

    // Render subscription tier UI elements
    const planName = document.getElementById('sidebar-plan-name');
    const planDesc = document.getElementById('sidebar-plan-desc');
    const planCard = document.getElementById('sidebar-plan-card');

    if (window.currentUser.role === 'Pro') {
        if (planName) planName.textContent = 'Pro Insight';
        if (planDesc) planDesc.textContent = 'Unlimited workspaces & API access.';
        if (planCard) planCard.className = "mb-4 p-4 bg-primary-container/20 border border-primary/40 rounded-lg pro-glow";
    } else {
        if (planName) planName.textContent = 'Free Plan';
        if (planDesc) planDesc.textContent = 'Unlock 10 projects & 4K ingestion.';
        if (planCard) planCard.className = "mb-4 p-4 bg-primary-container/10 border border-primary/20 rounded-lg";
    }
}

function handleLogout() {
    localStorage.removeItem('currentUser');
    window.currentUser = null;
    window.activeVideoId = null;
    navigate('login');
}

function showHelp() {
    alert("VideoInsight V2.0 Help Center:\n\n1. Ingestion Dashboard: Upload files directly or paste video urls (Google Drive / YouTube) to index transcripts & keyframes.\n2. Multimodal Chat: Ask queries about the active video stream, visual attributes or dialogue lines.");
}

// Hash-based route change listener
window.addEventListener('hashchange', () => {
    const page = window.location.hash.replace('#', '');
    if (page && page !== currentPage) {
        navigate(page);
    }
});

// Initialize application router on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const startPage = window.location.hash.replace('#', '') || 'dashboard';
    navigate(startPage);
});
