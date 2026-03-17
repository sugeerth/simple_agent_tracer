import { useState, useEffect, useCallback } from 'react';

const API = '/api/v1/auth';
const TOKEN_KEY = 'omniscope_token';
const USER_KEY = 'omniscope_user';

export interface User {
  user_id: string;
  username: string;
  email: string;
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(() => {
    const stored = localStorage.getItem(USER_KEY);
    return stored ? JSON.parse(stored) : null;
  });
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // On mount: validate stored token, or auto-enter demo mode on GitHub Pages
  useEffect(() => {
    const isGitHubPages = window.location.hostname.endsWith('.github.io');

    // Auto-enter demo mode on GitHub Pages (no backend available)
    if (!user && isGitHubPages) {
      const demoUser = { user_id: 'demo', username: 'demo', email: 'demo@omniscope.dev' };
      setUser(demoUser);
      setToken('demo');
      localStorage.setItem(USER_KEY, JSON.stringify(demoUser));
      localStorage.setItem(TOKEN_KEY, 'demo');
      return;
    }

    if (!token) return;
    fetch(`${API}/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          localStorage.removeItem(TOKEN_KEY);
          localStorage.removeItem(USER_KEY);
          setToken(null);
          setUser(null);
        }
      })
      .catch(() => { /* offline / demo mode - keep cached user */ });
  }, []);

  const signup = useCallback(async (email: string, username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, username, password }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return false;
      }
      localStorage.setItem(TOKEN_KEY, data.token);
      localStorage.setItem(USER_KEY, JSON.stringify({ user_id: data.user_id, username: data.username, email: data.email }));
      setToken(data.token);
      setUser({ user_id: data.user_id, username: data.username, email: data.email });
      return true;
    } catch {
      setError('Server unavailable. Try demo mode.');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const login = useCallback(async (usernameOrEmail: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username_or_email: usernameOrEmail, password }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return false;
      }
      localStorage.setItem(TOKEN_KEY, data.token);
      localStorage.setItem(USER_KEY, JSON.stringify({ user_id: data.user_id, username: data.username, email: data.email }));
      setToken(data.token);
      setUser({ user_id: data.user_id, username: data.username, email: data.email });
      return true;
    } catch {
      setError('Server unavailable. Try demo mode.');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    if (token) {
      try {
        await fetch(`${API}/logout`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch { /* ignore */ }
    }
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setToken(null);
    setUser(null);
  }, [token]);

  const enterDemoMode = useCallback(() => {
    const demoUser = { user_id: 'demo', username: 'demo', email: 'demo@omniscope.dev' };
    setUser(demoUser);
    setToken('demo');
    localStorage.setItem(USER_KEY, JSON.stringify(demoUser));
    localStorage.setItem(TOKEN_KEY, 'demo');
  }, []);

  return { user, token, loading, error, signup, login, logout, enterDemoMode };
}
