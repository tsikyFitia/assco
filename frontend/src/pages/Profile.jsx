// src/pages/Profile.jsx
import React from 'react'
import { useSelector } from 'react-redux'
import DashboardLayout from '../components/DashboardLayout'
import UserCard from '../components/UserCard'
import StatsCard from '../components/StatCards'
import ActionWidget from '../components/ActionWidget'

export default function Profile() {
  const { token, user } = useSelector((state) => state.auth)

  if (!token || !user) return <p>Non connecté</p>

  const stats = [
    { title: 'Projets', value: 8 },
    { title: 'Messages', value: 5 },
    { title: 'Notifications', value: 2 },
  ]

  return (
    <DashboardLayout>
      <h1 style={{ marginBottom: '2rem', textAlign: 'center' }}>Tableau de bord</h1>

      {/* Infos utilisateur */}
      <UserCard user={user} />

      {/* Statistiques */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '2rem' }}>
        {stats.map((s, idx) => (
          <StatsCard key={idx} title={s.title} value={s.value} />
        ))}
      </div>

      {/* Actions */}
      <ActionWidget />
    </DashboardLayout>
  )
}
