import React from 'react'
import { useSelector } from 'react-redux'
import { Link } from 'react-router-dom'
import LogoutButton from './LogoutButton'

// Material-UI imports
import { Drawer, List, ListItem, ListItemText, Divider, Typography, Box } from '@mui/material'

export default function Menu() {
  const user = useSelector((state) => state.auth.user)

  if (!user) return null

  const { role } = user

  const menus = {
    ADMIN: [
      { label: 'Dashboard Admin', path: '/admin' },
      { label: 'Utilisateurs', path: '/users' },
    ],
    STUDENT: [
      { label: 'Mes cours', path: '/courses' },
      { label: 'Mes notes', path: '/grades' },
    ],
    TEACHER: [
      { label: 'Mes classes', path: '/classes' },
    ],
    SCHOOL_ADMIN: [
      { label: 'Teachers', path: '/teacher' },
      { label: 'School', path: '/school' },
    ],
  }

  const roleMenus = menus[role] || []

  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: 240,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
          paddingTop: '1rem',
        },
      }}
    >
      <Box sx={{ textAlign: 'center', mb: 2 }}>
        <Typography variant="h6">{user.name || 'Utilisateur'}</Typography>
        <Typography variant="body2" color="textSecondary">{role}</Typography>
      </Box>

      <Divider />

      <List>
        {roleMenus.map((item) => (
          <ListItem button component={Link} to={item.path} key={item.path}>
            <ListItemText primary={item.label} />
          </ListItem>
        ))}
      </List>

      <Box sx={{ marginTop: 'auto', p: 2 }}>
        <LogoutButton
          style={{
            width: '100%',
            padding: '0.5rem',
            backgroundColor: '#e74c3c',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
          }}
        />
      </Box>
    </Drawer>
  )
}
