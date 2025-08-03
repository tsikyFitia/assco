import React from 'react'
import { useSelector } from 'react-redux'

export default function Profile() {
  const { token, user } = useSelector((state) => state.auth)

  // Ajout du console.log ici
  console.log('Token:', token)
  console.log('User:', user)

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Profil</h1>
      {token && user ? (
        <>
          <p>Bienvenue {user.name || user.email || 'utilisateur'} !</p>
          <p>Votre token : <code>{token}</code></p>
          {/* Tu peux afficher plus d'infos user ici */}
        </>
      ) : (
        <p>Non connecté</p>
      )}
    </div>
  )
}
