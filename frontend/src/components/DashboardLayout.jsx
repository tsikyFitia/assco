import React from 'react'
import Menu from './Menu'

import { Box, CssBaseline, Toolbar } from '@mui/material'

const drawerWidth = 240

export default function DashboardLayout({ children }) {
  return (
    <Box sx={{ display: 'flex' }}>
      {/* Reset CSS global */}
      <CssBaseline />

      {/* Sidebar */}
      <Menu />

      {/* Contenu principal */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3, // padding
          width: { sm: `calc(100% - ${drawerWidth}px)` },
        }}
      >
        {/* Toolbar pour compenser le Drawer si tu mets AppBar plus tard */}
        <Toolbar />
        {children}
      </Box>
    </Box>
  )
}