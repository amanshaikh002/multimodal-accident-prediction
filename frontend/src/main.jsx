import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './landing.css'
import Root from './Root.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
