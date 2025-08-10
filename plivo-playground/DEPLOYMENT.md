# 🚀 Deployment Guide - AI Playground

This guide provides step-by-step instructions for deploying the AI Playground application to various hosting platforms.

## 📋 Pre-deployment Checklist

1. ✅ Ensure all code is committed to a Git repository
2. ✅ Test the application locally (`npm start`)
3. ✅ Build the application successfully (`npm run build`)
4. ✅ Configure environment variables if using real APIs

## 🌐 Deployment Options

### 1. Netlify (Recommended for Quick Demo)

**Option A: Drag & Drop (Fastest)**
```bash
# Build the project
npm run build

# Go to netlify.com, drag the 'build' folder to deploy
```

**Option B: CLI Deployment**
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy using our custom script
npm run deploy:netlify
```

**Option C: Git Integration**
1. Push code to GitHub
2. Connect Netlify to your GitHub repository
3. Set build command: `npm run build`
4. Set publish directory: `build`
5. Deploy automatically on each push

### 2. Vercel

**Option A: CLI Deployment**
```bash
# Install Vercel CLI
npm install -g vercel

# Login and deploy
vercel login
npm run deploy:vercel
```

**Option B: Git Integration**
1. Push code to GitHub
2. Connect Vercel to your GitHub repository
3. Deploy automatically

### 3. GitHub Pages

```bash
# Update the homepage in package.json to your GitHub Pages URL
# Example: "homepage": "https://yourusername.github.io/plivo-playground"

# Deploy
npm run deploy

# Your app will be available at: https://yourusername.github.io/plivo-playground
```

### 4. AWS S3 + CloudFront

```bash
# Build the project
npm run build

# Create S3 bucket (replace with your bucket name)
aws s3 mb s3://your-ai-playground-bucket

# Configure bucket for static website hosting
aws s3 website s3://your-ai-playground-bucket --index-document index.html --error-document index.html

# Upload files
aws s3 sync build/ s3://your-ai-playground-bucket --delete

# Set up CloudFront distribution (optional, for better performance)
```

### 5. Firebase Hosting

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login

# Initialize Firebase in your project
firebase init hosting

# Build and deploy
npm run build
firebase deploy
```

## 🔧 Production Configuration

### Environment Variables

For production deployment with real APIs, create a `.env.production` file:

```env
REACT_APP_OPENAI_API_KEY=your_openai_api_key_here
REACT_APP_API_BASE_URL=https://your-backend-api.com
```

### API Integration Steps

1. **OpenAI Whisper API**:
   - Sign up for OpenAI API access
   - Get your API key
   - Uncomment the real API code in `src/services/TranscriptService.ts`
   - Add API key to environment variables

2. **CORS Configuration**:
   - If deploying a backend, configure CORS for your frontend domain
   - Add your deployment URL to allowed origins

## 🌟 Quick Demo Deployment (5 minutes)

For the fastest deployment to show your application:

```bash
# 1. Build the project
npm run build

# 2. Deploy to Netlify (drag & drop method)
# - Go to https://app.netlify.com/drop
# - Drag the 'build' folder to the deployment area
# - Get your live URL instantly!

# Alternative: Use Vercel
npx vercel --prod
```

## 🔐 Security Considerations

### For Production Deployment:

1. **API Keys**: Never expose API keys in frontend code
   - Use backend proxy for API calls
   - Implement proper authentication

2. **Authentication**: Replace demo auth with real authentication
   - Auth0, Firebase Auth, or custom backend
   - Implement proper session management

3. **File Upload**: Implement proper file validation
   - Server-side validation
   - File size limits
   - Virus scanning

## 📊 Performance Optimization

### Before Deployment:

```bash
# Analyze bundle size
npm install -g webpack-bundle-analyzer
npm run build
npx webpack-bundle-analyzer build/static/js/*.js

# Optimize images and assets
# Consider implementing lazy loading for large components
```

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails**:
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

2. **Routing Issues on Production**:
   - Configure your hosting to serve `index.html` for all routes
   - Add `_redirects` file for Netlify: `/* /index.html 200`

3. **CORS Errors**:
   - Ensure your backend allows requests from your deployed domain
   - Check browser console for specific CORS error details

## 📈 Monitoring and Analytics

### After Deployment:

1. **Add Google Analytics** (optional):
   - Create GA4 property
   - Add tracking code to `public/index.html`

2. **Error Tracking** (recommended):
   - Integrate Sentry or similar service
   - Monitor application errors in production

## 🎯 Demo URL Examples

After deployment, your application will be available at URLs like:

- **Netlify**: `https://magical-fairy-123456.netlify.app`
- **Vercel**: `https://plivo-playground-username.vercel.app`
- **GitHub Pages**: `https://username.github.io/plivo-playground`

## 📞 Support

If you encounter deployment issues:

1. Check the hosting platform's documentation
2. Review the application logs
3. Ensure all dependencies are installed
4. Verify environment variables are set correctly

---

**Ready to deploy? Choose your preferred method above and get your AI Playground live in minutes!** 🚀 