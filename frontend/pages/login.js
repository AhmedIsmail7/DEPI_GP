function renderLogin() {
    return `
    <main class="relative z-10 w-full max-w-md px-gutter animate-in fade-in duration-700 mx-auto">
        <!-- Brand Identity -->
        <div class="text-center mb-stack-lg">
            <h1 class="font-headline-lg text-headline-lg font-extrabold text-primary mb-2 tracking-tight">
                AIVision
            </h1>
            <p class="font-body-md text-body-md text-on-surface-variant">
                VideoInsight V2.0 Enterprise
            </p>
        </div>
        <!-- Login Card -->
        <div class="glass-panel p-stack-lg rounded-xl shadow-2xl">
            <div class="mb-stack-lg">
                <h2 class="font-headline-md text-headline-md font-semibold text-on-surface mb-1">Welcome back</h2>
                <p class="font-body-sm text-body-sm text-on-surface-variant">Access your AI analytics dashboard</p>
            </div>
            
            <div id="login-error" class="hidden p-3 mb-4 bg-error-container text-on-error-container rounded-lg text-xs font-medium"></div>

            <form class="space-y-stack-md" onsubmit="handleLoginSubmit(event)">
                <!-- Email Field -->
                <div class="space-y-stack-sm">
                    <label class="font-label-mono text-[11px] uppercase tracking-wider text-on-surface-variant" for="login-email">Email Address</label>
                    <div class="relative flex items-center bg-background rounded-lg border border-outline-variant transition-all duration-200 input-glow">
                        <span class="material-symbols-outlined absolute left-3 text-on-surface-variant text-lg">mail</span>
                        <input class="w-full bg-transparent border-none focus:ring-0 py-3 pl-10 pr-4 text-on-surface placeholder-outline font-body-sm rounded-lg" id="login-email" placeholder="name@company.com" required type="email"/>
                    </div>
                </div>
                <!-- Password Field -->
                <div class="space-y-stack-sm">
                    <div class="flex justify-between items-center">
                        <label class="font-label-mono text-[11px] uppercase tracking-wider text-on-surface-variant" for="login-password">Password</label>
                        <a class="font-body-sm text-body-sm text-primary hover:underline transition-all cursor-pointer">Forgot?</a>
                    </div>
                    <div class="relative flex items-center bg-background rounded-lg border border-outline-variant transition-all duration-200 input-glow">
                        <span class="material-symbols-outlined absolute left-3 text-on-surface-variant text-lg">lock</span>
                        <input class="w-full bg-transparent border-none focus:ring-0 py-3 pl-10 pr-12 text-on-surface placeholder-outline font-body-sm rounded-lg" id="login-password" placeholder="••••••••" required type="password"/>
                        <button class="absolute right-3 text-on-surface-variant hover:text-on-surface transition-colors" type="button" onclick="toggleLoginPassword()">
                            <span id="login-password-toggle-icon" class="material-symbols-outlined text-lg">visibility</span>
                        </button>
                    </div>
                </div>
                <!-- Remember Me -->
                <div class="flex items-center gap-2 py-2">
                    <input class="w-4 h-4 rounded border-outline-variant bg-background text-primary-container focus:ring-primary focus:ring-offset-background" id="remember" type="checkbox"/>
                    <label class="font-body-sm text-body-sm text-on-surface-variant select-none cursor-pointer" for="remember">Remember this device</label>
                </div>
                <!-- Submit Button -->
                <button id="login-submit-btn" class="w-full py-3 bg-primary-container text-on-primary-container font-headline-md text-body-md font-semibold rounded-xl transition-all duration-300 hover:scale-[1.02] active:scale-95 glow-active flex items-center justify-center gap-2 group" type="submit">
                    <span id="login-btn-text">Sign In</span>
                    <span class="material-symbols-outlined group-hover:translate-x-1 transition-transform">arrow_forward</span>
                </button>
            </form>
            <!-- Divider -->
            <div class="relative my-stack-lg">
                <div class="absolute inset-0 flex items-center">
                    <div class="w-full border-t border-outline-variant/30"></div>
                </div>
                <div class="relative flex justify-center text-xs">
                    <span class="px-2 bg-[#1E293B] text-on-surface-variant font-label-mono">OR CONTINUE WITH</span>
                </div>
            </div>
            <!-- Social Auth -->
            <div class="grid grid-cols-2 gap-stack-md">
                <button onclick="handleSocialLogin('Google')" class="flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg border border-outline-variant hover:bg-surface-container-highest transition-all duration-200">
                    <svg class="w-5 h-5 text-on-background" viewbox="0 0 24 24">
                        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="currentColor"></path>
                        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="currentColor"></path>
                        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="currentColor"></path>
                        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="currentColor"></path>
                    </svg>
                    <span class="font-body-sm text-body-sm">Google</span>
                </button>
                <button onclick="handleSocialLogin('GitHub')" class="flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg border border-outline-variant hover:bg-surface-container-highest transition-all duration-200">
                    <svg class="w-5 h-5 text-on-surface" fill="currentColor" viewbox="0 0 24 24">
                        <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"></path>
                    </svg>
                    <span class="font-body-sm text-body-sm">GitHub</span>
                </button>
            </div>
        </div>
        <!-- Footer Action -->
        <p class="text-center mt-stack-lg font-body-sm text-body-sm text-on-surface-variant">
            Don't have an account? 
            <a onclick="navigate('signup')" class="text-primary font-semibold hover:underline decoration-2 underline-offset-4 ml-1 cursor-pointer">Register here</a>
        </p>
        <!-- Technical Metadata Decoration -->
        <div class="mt-stack-lg pt-stack-lg flex flex-col items-center gap-2 opacity-30 select-none pointer-events-none">
            <div class="flex gap-4 font-code-sm text-xs text-primary uppercase tracking-[0.2em]">
                <span>Kernel: 0.12.4</span>
                <span>Region: US-EAST</span>
            </div>
            <div class="w-16 h-px bg-outline-variant"></div>
        </div>
    </main>
    `;
}

function toggleLoginPassword() {
    const passwordInput = document.getElementById('login-password');
    const toggleIcon = document.getElementById('login-password-toggle-icon');
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        toggleIcon.textContent = 'visibility_off';
    } else {
        passwordInput.type = 'password';
        toggleIcon.textContent = 'visibility';
    }
}

async function handleLoginSubmit(event) {
    event.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const errorDiv = document.getElementById('login-error');
    const submitBtn = document.getElementById('login-submit-btn');
    const btnText = document.getElementById('login-btn-text');

    errorDiv.classList.add('hidden');
    submitBtn.disabled = true;
    btnText.textContent = "Verifying Credentials...";

    try {
        const response = await fetch(`${window.API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('currentUser', JSON.stringify(data.user));
            window.currentUser = data.user;
            initSession();
            navigate('dashboard');
        } else {
            errorDiv.textContent = data.detail || "Authentication failed. Please verify your credentials.";
            errorDiv.classList.remove('hidden');
            submitBtn.disabled = false;
            btnText.textContent = "Sign In";
        }
    } catch (err) {
        console.error("Login error:", err);
        errorDiv.textContent = "Unable to connect to the authentication server. Please try again.";
        errorDiv.classList.remove('hidden');
        submitBtn.disabled = false;
        btnText.textContent = "Sign In";
    }
}

function handleSocialLogin(provider) {
    // Quick social login bypass for demo
    const mockUser = {
        email: `researcher@${provider.toLowerCase()}.com`,
        role: 'Pro',
        name: `Research Lead (${provider})`
    };
    localStorage.setItem('currentUser', JSON.stringify(mockUser));
    window.currentUser = mockUser;
    initSession();
    navigate('dashboard');
}
