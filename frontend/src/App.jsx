import React, { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { fetchMe } from './features/auth/authSlice'
import Menu from './components/Menu'
import Login from './pages/Login'
import Profile from './pages/Profile'
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom'

export default function App() {
  const dispatch = useDispatch()
  const { token, user, isAuthenticated, loading } = useSelector((state) => state.auth)

  // ➤ Charge les infos utilisateur si on a un token
  useEffect(() => {
    if (token && !user) {
      dispatch(fetchMe())
    }
  }, [token, user, dispatch])

  // ➤ Affiche une indication pendant le chargement
  if (loading) return <p style={{ textAlign: 'center', marginTop: '100px' }}>Chargement...</p>

  return (
    <Router>
      {/* ➤ Affiche le menu uniquement si l'utilisateur est connu */}
      {user && <Menu />}

      <Routes>
        {!isAuthenticated ? (
          <>
            <Route path="/login" element={<Login />} />
            <Route path="*" element={<Navigate to="/login" replace />} />
          </>
        ) : (
          <>
            <Route path="/profile" element={<Profile />} />
            <Route path="*" element={<Navigate to="/profile" replace />} />
          </>
        )}
      </Routes>
    </Router>
  )
}
