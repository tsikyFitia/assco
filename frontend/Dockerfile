# Utilise une version stable de Node compatible avec Vite
FROM node:20-alpine

# Dossier de travail
WORKDIR /app

# Copie les fichiers package.json et package-lock.json (si présent)
COPY package*.json ./

# Installe les dépendances (vite, react, etc.)
RUN npm install

# Copie le reste des fichiers (code source, vite.config.js, etc.)
COPY . .

# Expose le port utilisé par Vite
EXPOSE 5173

# Déclare l'adresse d’écoute (0.0.0.0 pour accès depuis l’hôte)
ENV HOST=0.0.0.0

# Démarre le serveur Vite
CMD ["npm", "run", "dev"]
