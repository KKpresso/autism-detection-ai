# Autism Detection AI Deployment Guide

This guide explains how to deploy the Autism Detection AI system using Docker and Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU deployment)
- At least 16GB RAM
- 50GB disk space
- CUDA-compatible GPU (optional, for GPU acceleration)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autism-detection-ai.git
cd autism-detection-ai
```

2. Place your trained model in the `models` directory:
```bash
mkdir -p models
cp path/to/your/model.pt models/best_model.pt
```

3. Start the services:

For CPU deployment:
```bash
docker-compose up -d
```

For GPU deployment:
```bash
DEPLOYMENT_TARGET=gpu CUDA_VISIBLE_DEVICES=0 docker-compose up -d
```

4. Verify the deployment:
```bash
curl http://localhost:8000/health
```

## Services

The deployment includes the following services:

1. **API Service (Port 8000)**
   - FastAPI application serving the model
   - Handles fMRI data preprocessing and predictions
   - Includes data quality control
   - Caches predictions for A/B testing

2. **Prometheus (Port 9090)**
   - Metrics collection and storage
   - Monitors API performance and model predictions
   - Tracks data quality and system health

3. **Grafana (Port 3000)**
   - Visualization dashboard for metrics
   - Default credentials: admin/admin
   - Pre-configured dashboards for model monitoring

## Configuration

### Environment Variables

- `DEPLOYMENT_TARGET`: Set to 'gpu' for GPU deployment, 'cpu' for CPU deployment
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices for deployment
- `GRAFANA_PASSWORD`: Set custom Grafana admin password
- `NUM_WORKERS`: Number of worker processes (default: 4)
- `MAX_CACHE_SIZE_GB`: Maximum cache size in GB (default: 50)

### Resource Requirements

Minimum requirements:
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB
- GPU: NVIDIA GPU with 8GB VRAM (for GPU deployment)

## API Endpoints

1. **Authentication**
   ```
   POST /token
   Content-Type: application/x-www-form-urlencoded
   ```
   Get access and refresh tokens.
   ```json
   {
     "username": "string",
     "password": "string"
   }
   ```

2. **Token Refresh**
   ```
   POST /refresh
   Authorization: Bearer <access_token>
   ```
   Refresh access token.

3. **Prediction**
   ```
   POST /predict
   Authorization: Bearer <access_token>
   Content-Type: multipart/form-data
   ```
   Upload fMRI data (NIfTI format) for prediction.

4. **User Info**
   ```
   GET /users/me
   Authorization: Bearer <access_token>
   ```
   Get current user information.

5. **User Predictions**
   ```
   GET /users/me/items
   Authorization: Bearer <access_token>
   ```
   Get user's prediction history.

6. **Health Check**
   ```
   GET /health
   ```
   Check API health and model version.

7. **Metrics**
   ```
   GET /metrics
   ```
   Get prediction metrics and system statistics.

## Authentication

The API supports both JWT token authentication and OAuth 2.0:

1. **OAuth Providers**:
   - Google
   - Microsoft
   - GitHub

2. **OAuth Setup**:

   a. **Google OAuth**:
   1. Go to [Google Cloud Console](https://console.cloud.google.com)
   2. Create a new project
   3. Enable OAuth 2.0 API
   4. Create OAuth credentials
   5. Add authorized redirect URI: `http://your-domain/auth/google/callback`
   6. Copy client ID and secret to `config/oauth_config.json`

   b. **Microsoft OAuth**:
   1. Go to [Azure Portal](https://portal.azure.com)
   2. Register a new application
   3. Add platform: Web
   4. Add redirect URI: `http://your-domain/auth/microsoft/callback`
   5. Create client secret
   6. Copy client ID and secret to `config/oauth_config.json`

   c. **GitHub OAuth**:
   1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
   2. Create new OAuth App
   3. Add homepage URL and callback URL
   4. Copy client ID and secret to `config/oauth_config.json`

3. **OAuth Configuration**:
   ```json
   {
     "provider_name": {
       "client_id": "YOUR_CLIENT_ID",
       "client_secret": "YOUR_CLIENT_SECRET",
       "redirect_uri": "http://your-domain/auth/provider/callback"
     }
   }
   ```

4. **OAuth Endpoints**:
   ```
   # Start OAuth flow
   GET /auth/{provider}/login
   
   # OAuth callback
   GET /auth/{provider}/callback
   
   # List providers
   GET /auth/providers
   ```

5. **OAuth Flow**:
   1. User visits `/auth/{provider}/login`
   2. User is redirected to provider login
   3. After login, provider redirects to callback URL
   4. API creates/updates user and returns tokens
   5. Use tokens for subsequent API calls

## Monitoring

1. Access Prometheus:
   ```
   http://localhost:9090
   ```

2. Access Grafana:
   ```
   http://localhost:3000
   ```

Key metrics monitored:
- Prediction latency
- QC failure rate
- Model performance
- System resources
- Cache utilization

## Scaling

To scale the API service:
```bash
docker-compose up -d --scale api=3
```

## Troubleshooting

1. **API Service Issues**
   - Check logs: `docker-compose logs api`
   - Verify model file exists
   - Check resource usage

2. **GPU Issues**
   - Verify NVIDIA runtime: `nvidia-smi`
   - Check CUDA compatibility
   - Verify GPU access rights

3. **Memory Issues**
   - Adjust cache size
   - Monitor memory usage
   - Consider scaling resources

## Maintenance

1. **Updating the Model**
   ```bash
   # Stop services
   docker-compose down
   
   # Replace model file
   cp new_model.pt models/best_model.pt
   
   # Restart services
   docker-compose up -d
   ```

2. **Clearing Cache**
   ```bash
   # Remove cache directory
   rm -rf cache/*
   
   # Restart API service
   docker-compose restart api
   ```

3. **Backup**
   ```bash
   # Backup model and cache
   tar -czf backup.tar.gz models/ cache/
   ```

## Security Considerations

1. **API Security**
   - Enable CORS restrictions
   - Add authentication
   - Use HTTPS in production

2. **Data Security**
   - Encrypt sensitive data
   - Regular backups
   - Access control

3. **System Security**
   - Keep containers updated
   - Monitor system access
   - Regular security audits

## Support

For issues and support:
1. Check the logs: `docker-compose logs`
2. Review error messages
3. Contact the development team

## License

This project is licensed under [Your License]. See LICENSE file for details.
