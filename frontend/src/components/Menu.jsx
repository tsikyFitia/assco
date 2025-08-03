import { useSelector } from 'react-redux'
import { Link } from 'react-router-dom'

export default function Menu() {
  const user = useSelector((state) => state.auth.user)

  // Si pas d'utilisateur connecté, ne pas afficher le menu
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
  }

  // Récupérer le menu correspondant au rôle, ou un tableau vide sinon
  const roleMenus = menus[role] || []

  return (
    <ul>
      {roleMenus.map((item) => (
        <li key={item.path}>
          <Link to={item.path}>{item.label}</Link>
        </li>
      ))}
    </ul>
  )
}
