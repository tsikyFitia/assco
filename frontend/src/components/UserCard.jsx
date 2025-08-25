// src/components/UserCard.jsx
import React from 'react'

export default function UserCard({ user }) {
  return (
    <div style={{
      padding: '1.5rem',
      borderRadius: '10px',
      boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
      backgroundColor: '#fff',
      marginBottom: '1rem',
    }}>
      <h2 style={{ marginBottom: '0.5rem' }}>{user.name || user.email}</h2>
      <p><strong>Email:</strong> {user.email}</p>
      <p><strong>Role:</strong> {user.role || 'Utilisateur'}</p>
    </div>
  )
}
