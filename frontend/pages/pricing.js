function renderPricing() {
    const isPro = window.currentUser && window.currentUser.role === 'Pro';
    return `
    <!-- Main Content Canvas -->
    <main class="flex-grow pt-8 pb-12 px-gutter relative overflow-y-auto h-full scrollbar-hide">
        <!-- Atmospheric Background Element -->
        <div class="absolute top-0 right-0 w-1/2 h-1/2 bg-primary/5 blur-[120px] rounded-full pointer-events-none"></div>
        
        <div class="max-w-5xl mx-auto flex flex-col items-center text-center mb-10 relative z-10">
            <span class="font-label-mono text-label-mono uppercase tracking-widest text-secondary mb-4">VideoInsight V2.0 Plans</span>
            <h1 class="font-display-lg text-[32px] md:text-[40px] text-on-surface mb-4 leading-tight">Calculated Power for Video Intelligence</h1>
            <p class="font-body-lg text-body-md text-on-surface-variant max-w-2xl">
                Harness high-density information processing with precision. Choose the plan that scales with your engineering requirements.
            </p>
        </div>
        
        <!-- Pricing Grid -->
        <div class="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 px-container-padding relative z-10">
            <!-- Free Plan Card -->
            <div class="glass-card rounded-2xl p-8 flex flex-col hover:scale-[1.01] transition-transform duration-300 ${!isPro ? 'border-primary/20 ring-1 ring-primary/10' : 'border-outline-variant/30'}">
                <div class="mb-8">
                    <h2 class="font-headline-md text-headline-md text-on-surface mb-2">Free Plan</h2>
                    <div class="flex items-baseline gap-1">
                        <span class="font-display-lg text-[36px] font-bold text-on-surface">$0</span>
                        <span class="text-on-surface-variant">/mo</span>
                    </div>
                    <p class="text-[13px] text-on-surface-variant mt-2">Essential tools for individual researchers.</p>
                </div>
                <div class="space-y-4 mb-10 flex-grow">
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md">check_circle</span>
                        <span class="text-on-surface text-[14px]">60 min videos</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md">check_circle</span>
                        <span class="text-on-surface text-[14px]">5 projects limit</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md">check_circle</span>
                        <span class="text-on-surface text-[14px]">Basic AI transcription</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-outline text-body-md">block</span>
                        <span class="text-on-surface-variant text-[14px] line-through opacity-50">API Access</span>
                    </div>
                </div>
                <button onclick="selectPlan('Free')" class="w-full py-4 rounded-xl border border-outline-variant text-on-surface font-semibold hover:bg-surface-variant transition-colors active:scale-[0.98]">
                    ${!isPro ? 'Current Plan' : 'Downgrade to Free'}
                </button>
            </div>
            
            <!-- Pro Insight Plan Card -->
            <div class="glass-card rounded-2xl p-8 flex flex-col relative pro-glow hover:scale-[1.01] transition-transform duration-300 ${isPro ? 'border-primary ring-1 ring-primary/30' : 'border-primary/40 ring-1 ring-primary/20'}">
                <div class="absolute -top-4 left-1/2 -translate-x-1/2 bg-primary px-4 py-1 rounded-full shadow-lg">
                    <span class="font-label-mono text-[10px] text-on-primary font-bold uppercase tracking-tighter">Recommended</span>
                </div>
                <div class="mb-8">
                    <h2 class="font-headline-md text-headline-md text-primary mb-2">Pro Insight Plan</h2>
                    <div class="flex items-baseline gap-1">
                        <span class="font-display-lg text-[36px] font-bold text-on-surface">$19</span>
                        <span class="text-on-surface-variant">/mo</span>
                    </div>
                    <p class="text-[13px] text-on-surface-variant mt-2">High-performance features for technical leads.</p>
                </div>
                <div class="space-y-4 mb-10 flex-grow">
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md" style="font-variation-settings: 'FILL' 1;">check_circle</span>
                        <span class="text-on-surface font-medium text-[14px]">180 min videos</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md" style="font-variation-settings: 'FILL' 1;">check_circle</span>
                        <span class="text-on-surface font-medium text-[14px]">Unlimited workspaces</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md" style="font-variation-settings: 'FILL' 1;">check_circle</span>
                        <span class="text-on-surface font-medium text-[14px]">Full Multimodal AI Chat</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined text-primary text-body-md" style="font-variation-settings: 'FILL' 1;">check_circle</span>
                        <span class="text-on-surface font-medium text-[14px]">Complete API Access</span>
                    </div>
                </div>
                <button onclick="selectPlan('Pro')" class="w-full py-4 bg-primary-container text-on-primary-container rounded-xl font-bold hover:bg-primary transition-all active:scale-[0.98]">
                    ${isPro ? 'Current Plan' : 'Upgrade to Pro'}
                </button>
            </div>
        </div>
    </main>
    `;
}

function selectPlan(plan) {
    if (!window.currentUser) {
        navigate('login');
        return;
    }
    window.currentUser.role = plan;
    localStorage.setItem('currentUser', JSON.stringify(window.currentUser));
    
    // Update interface dynamically
    const badge = document.getElementById('user-role-badge');
    if (badge) {
        badge.textContent = plan;
    }
    
    // Update sidebar card
    const planName = document.getElementById('sidebar-plan-name');
    const planDesc = document.getElementById('sidebar-plan-desc');
    const planCard = document.getElementById('sidebar-plan-card');
    
    if (plan === 'Pro') {
        if (planName) planName.textContent = 'Pro Insight';
        if (planDesc) planDesc.textContent = 'Unlimited workspaces & API access.';
    } else {
        if (planName) planName.textContent = 'Free Plan';
        if (planDesc) planDesc.textContent = 'Unlock 10 projects & 4K ingestion.';
    }
    
    // Re-render
    navigate('pricing');
}
