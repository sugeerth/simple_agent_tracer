import React, { useState } from 'react';

interface Props {
  onLogin: (usernameOrEmail: string, password: string) => Promise<boolean>;
  onSignup: (email: string, username: string, password: string) => Promise<boolean>;
  onDemoMode: () => void;
  loading: boolean;
  error: string | null;
}

export default function LoginPage({ onLogin, onSignup, onDemoMode, loading, error }: Props) {
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (mode === 'signup') {
      if (password !== confirmPassword) {
        setLocalError('Passwords do not match');
        return;
      }
      if (password.length < 6) {
        setLocalError('Password must be at least 6 characters');
        return;
      }
      await onSignup(email, username, password);
    } else {
      await onLogin(username || email, password);
    }
  };

  const displayError = localError || error;

  return (
    <div className="login-page">
      <div className="login-bg-grid" />
      <div className="login-container">
        <div className="login-logo">
          <h1>OMNISCOPE</h1>
          <p className="login-tagline">Multi-Agent Observability Platform</p>
        </div>

        <div className="login-card">
          <div className="login-tabs">
            <button
              className={`login-tab ${mode === 'login' ? 'active' : ''}`}
              onClick={() => { setMode('login'); setLocalError(null); }}
            >
              Sign In
            </button>
            <button
              className={`login-tab ${mode === 'signup' ? 'active' : ''}`}
              onClick={() => { setMode('signup'); setLocalError(null); }}
            >
              Create Account
            </button>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            {mode === 'signup' && (
              <div className="login-field">
                <label>Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  autoComplete="email"
                />
              </div>
            )}

            <div className="login-field">
              <label>{mode === 'login' ? 'Username or Email' : 'Username'}</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder={mode === 'login' ? 'username or email' : 'username'}
                required
                autoComplete="username"
              />
            </div>

            <div className="login-field">
              <label>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="password"
                required
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
            </div>

            {mode === 'signup' && (
              <div className="login-field">
                <label>Confirm Password</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  placeholder="confirm password"
                  required
                  autoComplete="new-password"
                />
              </div>
            )}

            {displayError && (
              <div className="login-error">{displayError}</div>
            )}

            <button type="submit" className="login-submit" disabled={loading}>
              {loading ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
            </button>
          </form>

          <div className="login-divider">
            <span>or</span>
          </div>

          <button className="login-demo" onClick={onDemoMode}>
            Explore Demo Mode
          </button>

          <div className="login-features">
            <div className="login-feature">
              <span className="feature-icon">&#9878;</span>
              <span>Agent DAG Tracing</span>
            </div>
            <div className="login-feature">
              <span className="feature-icon">&#9733;</span>
              <span>Multi-LLM Judge Panel</span>
            </div>
            <div className="login-feature">
              <span className="feature-icon">&#9888;</span>
              <span>Predictive Failure Detection</span>
            </div>
            <div className="login-feature">
              <span className="feature-icon">&#8634;</span>
              <span>Time-Travel Debugging</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
