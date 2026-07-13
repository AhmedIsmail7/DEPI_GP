function renderSignup() {
    return `
    <main class="min-h-screen flex items-center justify-center p-gutter relative overflow-hidden mx-auto w-full max-w-lg z-10">
        <div class="w-full">
            <!-- Brand Identity -->
            <div class="flex flex-col items-center mb-stack-lg animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div class="w-16 h-16 bg-primary-container rounded-xl flex items-center justify-center mb-stack-md glow-accent">
                    <span class="material-symbols-outlined text-on-primary-container text-4xl" style="font-variation-settings: 'FILL' 1;">visibility</span>
                </div>
                <h1 class="font-headline-lg text-headline-lg text-primary tracking-tight">VideoInsight <span class="text-on-surface-variant font-normal">V2.0</span></h1>
                <p class="font-body-md text-body-md text-on-surface-variant mt-2">Precision AI Analysis for Developers</p>
            </div>
            
            <!-- Sign Up Card -->
            <div class="glass-card p-stack-lg rounded-xl shadow-2xl animate-in fade-in zoom-in-95 duration-500 delay-150">
                <div id="signup-error" class="hidden p-3 mb-4 bg-error-container text-on-error-container rounded-lg text-xs font-medium"></div>
                <div id="signup-success" class="hidden p-3 mb-4 bg-green-500/20 border border-green-500/30 text-green-400 rounded-lg text-xs font-medium"></div>

                <form class="space-y-stack-md" onsubmit="handleSignupSubmit(event)">
                    <div class="grid grid-cols-1 gap-stack-md">
                        <!-- Full Name -->
                        <div class="space-y-unit">
                            <label class="font-label-mono text-outline text-[11px] uppercase tracking-wider block ml-1">Full Name</label>
                            <div class="relative group flex items-center bg-surface border border-outline-variant rounded-lg">
                                <span class="material-symbols-outlined absolute left-3 text-outline-variant transition-colors">person</span>
                                <input id="signup-name" class="w-full bg-transparent border-none focus:ring-0 py-3 pl-11 pr-4 font-body-md text-body-md text-on-surface focus:outline-none transition-all rounded-lg" placeholder="Alex Sterling" type="text" required/>
                            </div>
                        </div>
                        <!-- Email -->
                        <div class="space-y-unit">
                            <label class="font-label-mono text-outline text-[11px] uppercase tracking-wider block ml-1">Email Address</label>
                            <div class="relative group flex items-center bg-surface border border-outline-variant rounded-lg">
                                <span class="material-symbols-outlined absolute left-3 text-outline-variant transition-colors">mail</span>
                                <input id="signup-email" class="w-full bg-transparent border-none focus:ring-0 py-3 pl-11 pr-4 font-body-md text-body-md text-on-surface focus:outline-none transition-all rounded-lg" placeholder="alex@tech-vision.io" type="email" required/>
                            </div>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-stack-md">
                            <!-- Password -->
                            <div class="space-y-unit">
                                <label class="font-label-mono text-outline text-[11px] uppercase tracking-wider block ml-1">Password</label>
                                <div class="relative group flex items-center bg-surface border border-outline-variant rounded-lg">
                                    <span class="material-symbols-outlined absolute left-3 text-outline-variant transition-colors">lock</span>
                                    <input id="signup-password" class="w-full bg-transparent border-none focus:ring-0 py-3 pl-11 pr-4 font-body-md text-body-md text-on-surface focus:outline-none transition-all rounded-lg" placeholder="••••••••" type="password" required/>
                                </div>
                            </div>
                            <!-- Confirm Password -->
                            <div class="space-y-unit">
                                <label class="font-label-mono text-outline text-[11px] uppercase tracking-wider block ml-1">Confirm</label>
                                <div class="relative group flex items-center bg-surface border border-outline-variant rounded-lg">
                                    <span class="material-symbols-outlined absolute left-3 text-outline-variant transition-colors">verified_user</span>
                                    <input id="signup-confirm" class="w-full bg-transparent border-none focus:ring-0 py-3 pl-11 pr-4 font-body-md text-body-md text-on-surface focus:outline-none transition-all rounded-lg" placeholder="••••••••" type="password" required/>
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- Terms Checkbox -->
                    <div class="flex items-start gap-3 py-2">
                        <input class="mt-1 rounded border-outline-variant bg-surface text-primary-container focus:ring-primary-container/50" id="terms" type="checkbox" required/>
                        <label class="font-body-sm text-[12px] text-on-surface-variant leading-tight cursor-pointer" for="terms">
                            I agree to the <a class="text-primary hover:underline transition-all">Terms of Service</a> and acknowledge the <a class="text-primary hover:underline transition-all">Privacy Policy</a>.
                        </label>
                    </div>
                    <!-- CTA Button -->
                    <button id="signup-submit-btn" class="w-full bg-primary-container hover:bg-primary text-on-primary-container font-headline-md text-body-md py-4 rounded-xl shadow-lg transition-all active:scale-[0.98] flex items-center justify-center gap-2 glow-accent mt-stack-md group" type="submit">
                        <span id="signup-btn-text">Initialize Account</span>
                        <span class="material-symbols-outlined group-hover:translate-x-1 transition-transform">arrow_forward</span>
                    </button>
                </form>
                <!-- Footer Divider -->
                <div class="relative my-stack-lg">
                    <div class="absolute inset-0 flex items-center"><div class="w-full border-t border-outline-variant/30"></div></div>
                    <div class="relative flex justify-center text-xs uppercase"><span class="bg-surface-container px-2 text-outline font-label-mono">Or Register With</span></div>
                </div>
                <!-- Social Logins -->
                <div class="grid grid-cols-2 gap-stack-md">
                    <button onclick="handleSocialLogin('GitHub')" class="flex items-center justify-center gap-2 py-3 px-4 border border-outline-variant rounded-lg font-body-sm text-body-sm text-on-surface hover:bg-surface-container-high transition-colors active:scale-95">
                        <span class="material-symbols-outlined text-lg">terminal</span>
                        <span>GitHub</span>
                    </button>
                    <button onclick="handleSocialLogin('Google')" class="flex items-center justify-center gap-2 py-3 px-4 border border-outline-variant rounded-lg font-body-sm text-body-sm text-on-surface hover:bg-surface-container-high transition-colors active:scale-95">
                        <span class="material-symbols-outlined text-lg">cloud</span>
                        <span>Google</span>
                    </button>
                </div>
            </div>
            <!-- Bottom Link -->
            <div class="mt-stack-lg text-center animate-in fade-in duration-1000 delay-500">
                <p class="font-body-md text-body-md text-on-surface-variant">
                    Already part of the network? 
                    <a onclick="navigate('login')" class="text-primary font-semibold hover:underline transition-all ml-1 cursor-pointer">Authenticate here</a>
                </p>
            </div>
        </div>
    </main>
    `;
}

async function handleSignupSubmit(event) {
    event.preventDefault();
    const name = document.getElementById('signup-name').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const confirm = document.getElementById('signup-confirm').value;
    const errorDiv = document.getElementById('signup-error');
    const successDiv = document.getElementById('signup-success');
    const submitBtn = document.getElementById('signup-submit-btn');
    const btnText = document.getElementById('signup-btn-text');

    errorDiv.classList.add('hidden');
    successDiv.classList.add('hidden');

    if (password !== confirm) {
        errorDiv.textContent = "Passwords do not match. Please verify.";
        errorDiv.classList.remove('hidden');
        return;
    }

    submitBtn.disabled = true;
    btnText.textContent = "Creating Account...";

    try {
        const response = await fetch(`${window.API_BASE_URL}/api/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, name })
        });

        const data = await response.json();
        
        if (response.ok) {
            successDiv.textContent = "Account successfully initialized! Redirecting to login...";
            successDiv.classList.remove('hidden');
            setTimeout(() => {
                navigate('login');
            }, 1500);
        } else {
            errorDiv.textContent = data.detail || "Registration failed. Please try a different email.";
            errorDiv.classList.remove('hidden');
            submitBtn.disabled = false;
            btnText.textContent = "Initialize Account";
        }
    } catch (err) {
        console.error("Signup error:", err);
        errorDiv.textContent = "Unable to connect to the registration server. Please try again.";
        errorDiv.classList.remove('hidden');
        submitBtn.disabled = false;
        btnText.textContent = "Initialize Account";
    }
}
