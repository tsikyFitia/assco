// src/components/StatsCard.jsx
import React from 'react'

export default function StatsCard({ title, value }) {
  return (
    <div>
      <h3 style={{ marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{value}</p>
    </div>
  )
}
