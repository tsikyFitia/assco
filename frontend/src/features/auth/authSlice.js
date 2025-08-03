// src/features/auth/authSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

const API_URL = import.meta.env.VITE_API_URL

// Fonction utilitaire pour décoder un JWT (sans dépendance)
function parseJwt(token) {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    )
    return JSON.parse(jsonPayload)
  } catch {
    return null
  }
}

// 🔐 Requête de login
export const loginUser = createAsyncThunk(
  'auth/loginUser',
  async ({ email, password }, { rejectWithValue }) => {
    try {
      const response = await fetch(`${API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        return rejectWithValue(errorData.detail || 'Erreur de connexion')
      }

      const data = await response.json()
      localStorage.setItem('access_token', data.access_token)
      return data.access_token
    } catch (error) {
      return rejectWithValue(error.message)
    }
  }
)

// 👤 Requête pour récupérer les infos de l'utilisateur connecté
export const fetchMe = createAsyncThunk(
  'auth/fetchMe',
  async (_, { getState, rejectWithValue, dispatch }) => {
    try {
      const token = getState().auth.token || localStorage.getItem('access_token')
      console.log("🔐 Token envoyé à /me:", token)

      // Vérification expiration du token avant la requête
      const payload = parseJwt(token)
      if (payload && payload.exp * 1000 < Date.now()) {
        // Token expiré → déconnexion auto
        dispatch(logout())
        return rejectWithValue('Token expiré, veuillez vous reconnecter.')
      }

      const response = await fetch(`${API_URL}/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })

      if (response.status === 401) {
        // Token invalide / expiré côté serveur → déconnexion auto
        dispatch(logout())
        return rejectWithValue('Non autorisé, veuillez vous reconnecter.')
      }

      if (!response.ok) {
        return rejectWithValue('Impossible de récupérer les informations utilisateur')
      }

      const data = await response.json()
      return data
    } catch (error) {
      return rejectWithValue(error.message)
    }
  }
)

const initialState = {
  token: localStorage.getItem('access_token'),
  user: null,
  isAuthenticated: !!localStorage.getItem('access_token'),
  loading: false,
  error: null,
}

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    logout(state) {
      state.token = null
      state.user = null
      state.isAuthenticated = false
      state.error = null
      localStorage.removeItem('access_token')
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(loginUser.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.loading = false
        state.token = action.payload
        state.isAuthenticated = true
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload
        state.isAuthenticated = false
      })

      .addCase(fetchMe.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchMe.fulfilled, (state, action) => {
        console.log("✅ fetchMe SUCCESS:", action.payload)
        state.loading = false
        state.user = action.payload
      })
      .addCase(fetchMe.rejected, (state, action) => {
        console.error("❌ fetchMe ERROR:", action.payload)
        state.loading = false
        state.error = action.payload
        state.user = null
      })
  },
})

export const { logout } = authSlice.actions
export default authSlice.reducer
