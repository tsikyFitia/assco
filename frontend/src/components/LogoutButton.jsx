// src/components/LogoutButton.jsx
import React from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { logout } from '../features/auth/authSlice'
import { useNavigate } from 'react-router-dom'

export default function LogoutButton({ style, className }) {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const { isAuthenticated } = useSelector((state) => state.auth)

  const handleLogout = () => {
    dispatch(logout())
    navigate('/login')
  }

  // Ne rien afficher si l'utilisateur n'est pas connecté
  if (!isAuthenticated) return null

  return (
    <button
      onClick={handleLogout}
      style={style}
      className={className}
    >
      Logout
    </button>
  )
}
