# 🚀 Vercel Deployment Guide

This guide will help you deploy the AI Playground to Vercel for free hosting.

## Prerequisites

- [Node.js](https://nodejs.org/) (v14 or higher)
- [Git](https://git-scm.com/)
- [Vercel account](https://vercel.com/) (free)

## Quick Deployment Methods

### Method 1: Deploy via Vercel CLI (Recommended)

1. **Install Vercel CLI globally:**
   ```bash
   npm install -g vercel
   ```

2. **Navigate to the project directory:**
   ```bash
   cd plivo-playground
   ```

3. **Build the project:**
   ```bash
   npm run build
   ```

4. **Deploy to Vercel:**
   ```bash
   vercel
   ```
   
   Follow the prompts:
   - Login to your Vercel account
   - Confirm project settings
   - Choose deployment name

5. **For production deployment:**
   ```bash
   vercel --prod
   ```

### Method 2: Deploy via Vercel Dashboard

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add AI Playground"
   git push origin main
   ```

2. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**

3. **Click "New Project"**

4. **Import your GitHub repository**

5. **Configure build settings:**
   - **Framework Preset:** Create React App
   - **Build Command:** `npm run build`
   - **Output Directory:** `build`
   - **Install Command:** `npm install`

6. **Deploy!**

### Method 3: One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/ai-playground)

## Environment Variables

For production, you may want to set these environment variables in Vercel:

```
REACT_APP_API_BASE=https://your-backend-url.com
```

## Build Configuration

The project includes a `vercel.json` file with optimal settings:

```json
{
  "version": 2,
  "name": "ai-playground",
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "headers": {
        "cache-control": "s-maxage=31536000,immutable"
      },
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

## Custom Domain (Optional)

1. Go to your project settings in Vercel
2. Navigate to "Domains"
3. Add your custom domain
4. Follow DNS configuration instructions

## Monitoring and Analytics

Vercel provides built-in:
- **Performance monitoring**
- **Analytics dashboard**
- **Real-time logs**
- **Edge network optimization**

## Backend Deployment

The frontend will work with the demo mode, but for full functionality, deploy the backend:

### Option 1: Railway (Recommended for Python)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Navigate to backend
cd ../audio-backend

# Deploy
railway login
railway init
railway up
```

### Option 2: Render
1. Create account at [Render](https://render.com)
2. Connect GitHub repository
3. Choose "Web Service"
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option 3: Heroku
```bash
# Install Heroku CLI
# Create Procfile in audio-backend/:
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create your-ai-backend
git subtree push --prefix audio-backend heroku main
```

## Troubleshooting

### Build Errors
- Check Node.js version: `node --version` (should be 14+)
- Clear cache: `npm ci`
- Verify dependencies: `npm install`

### Route Issues
- Vercel automatically handles SPA routing with the `vercel.json` config
- All routes redirect to `index.html` for client-side routing

### Large Bundle Size
- The build is optimized automatically
- Static assets are cached with long-term headers
- Code splitting is handled by Create React App

## Performance Optimization

Vercel provides:
- **Global CDN** - Fast loading worldwide
- **Edge caching** - Static assets cached globally
- **Automatic compression** - Gzip/Brotli compression
- **Image optimization** - Automatic image optimization

## Support

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Community](https://vercel.com/community)
- [Create React App on Vercel](https://vercel.com/guides/deploying-react-with-vercel)

---

## 🎉 Your AI Playground is now live on Vercel!

After deployment, you'll get a URL like: `https://your-project-name.vercel.app`

Share this URL with others to showcase your multi-modal AI playground! 🚀 