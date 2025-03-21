# Mini-RAG Azure Migration Guide

## Overview

This document provides a comprehensive implementation guide for migrating the Mini-RAG system from local CPU-based deployment to Azure cloud services. The migration pathway preserves application functionality while leveraging Azure's managed services for improved scalability and resource optimization.

## Migration Architecture

The migration follows a structured service replacement strategy:

| Local Component | Azure Replacement | Migration Complexity |
|-----------------|-------------------|----------------------|
| LLM Service (phi-2.gguf) | Azure OpenAI | Medium |
| Embedding Service (sentence-transformers) | Azure OpenAI Embeddings | Low |
| Vector Store (ChromaDB) | Azure Cognitive Search | Medium |
| Document Storage (Local Filesystem) | Azure Blob Storage | Low |
| Backend API (FastAPI) | Azure Container Apps | Low |
| Frontend (React) | Azure Container Apps/Static Web Apps | Low |

## Prerequisites

- Azure subscription with appropriate resource provider permissions
- Azure CLI (version 2.45.0+) installed and authenticated
- Docker CLI (version 20.10+) for container operations
- Git repository access for application codebase
- Administrative access to deployment environment

## Implementation Process

### Phase 1: Azure Resource Provisioning

1. **Create Resource Group:**
   ```bash
   az group create --name mini-rag-rg --location eastus
   ```

2. **Deploy Azure OpenAI:**
   ```bash
   az cognitiveservices account create \
     --name mini-rag-openai \
     --resource-group mini-rag-rg \
     --location eastus \
     --kind OpenAI \
     --sku S0
   
   # Deploy model
   az cognitiveservices account deployment create \
     --name mini-rag-openai \
     --resource-group mini-rag-rg \
     --deployment-name gpt-4-turbo \
     --model-name gpt-4-turbo \
     --model-version latest \
     --model-format OpenAI \
     --sku Standard
   ```

3. **Deploy Azure Cognitive Search:**
   ```bash
   az search service create \
     --name mini-rag-search \
     --resource-group mini-rag-rg \
     --location eastus \
     --sku Basic
   ```

4. **Create Azure Storage Account:**
   ```bash
   az storage account create \
     --name miniragstore \
     --resource-group mini-rag-rg \
     --location eastus \
     --sku Standard_LRS

   # Create container for documents
   az storage container create \
     --name documents \
     --account-name miniragstore
   ```

5. **Create Azure Container Registry:**
   ```bash
   az acr create \
     --name miniragregistry \
     --resource-group mini-rag-rg \
     --location eastus \
     --sku Basic \
     --admin-enabled true
   ```

### Phase 2: Configuration Preparation

1. **Retrieve Service Keys and Endpoints:**
   ```bash
   # Retrieve OpenAI key
   OPENAI_KEY=$(az cognitiveservices account keys list \
     --name mini-rag-openai \
     --resource-group mini-rag-rg \
     --query key1 -o tsv)
   
   # Retrieve Search key
   SEARCH_KEY=$(az search admin-key show \
     --service-name mini-rag-search \
     --resource-group mini-rag-rg \
     --query primaryKey -o tsv)
   
   # Get storage connection string
   STORAGE_CONNECTION=$(az storage account show-connection-string \
     --name miniragstore \
     --resource-group mini-rag-rg \
     --query connectionString -o tsv)
   
   # Get container registry credentials
   ACR_SERVER=$(az acr show --name miniragregistry --query loginServer -o tsv)
   ACR_USERNAME=$(az acr credential show --name miniragregistry --query username -o tsv)
   ACR_PASSWORD=$(az acr credential show --name miniragregistry --query passwords[0].value -o tsv)
   ```

2. **Configure Environment Variables:**
   Create `.env.azure` file with the following structure:
   ```
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=https://mini-rag-openai.openai.azure.com
   AZURE_OPENAI_API_KEY=${OPENAI_KEY}
   AZURE_OPENAI_DEPLOYMENT=gpt-4-turbo
   
   # Azure Cognitive Search Configuration
   AZURE_SEARCH_ENDPOINT=https://mini-rag-search.search.windows.net
   AZURE_SEARCH_API_KEY=${SEARCH_KEY}
   AZURE_SEARCH_INDEX=documents
   
   # Azure Storage Configuration
   AZURE_STORAGE_CONNECTION_STRING=${STORAGE_CONNECTION}
   
   # Docker Registry Configuration
   DOCKER_REGISTRY_SERVER=${ACR_SERVER}
   DOCKER_USERNAME=${ACR_USERNAME}
   DOCKER_PASSWORD=${ACR_PASSWORD}
   
   # Project Configuration
   PROJECT_NAME=mini-rag
   IMAGE_TAG=v1.0
   ```

### Phase 3: Data Migration

1. **Document Migration Script:**
   Create a migration script `scripts/migrate_documents_to_azure.sh`:
   ```bash
   #!/bin/bash
   
   # Load environment variables
   source .env.azure
   
   # Directory containing documents
   DOCUMENTS_DIR="./data/documents"
   
   # Migrate each document to Azure Blob Storage
   find "$DOCUMENTS_DIR" -type f | while read -r file; do
     filename=$(basename "$file")
     echo "Uploading $filename to Azure Blob Storage..."
     
     # Upload file to Azure Blob Storage
     az storage blob upload \
       --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
       --container-name documents \
       --name "$filename" \
       --file "$file" \
       --overwrite
   done
   
   echo "Document migration complete."
   ```

2. **Execute Migration:**
   ```bash
   chmod +x scripts/migrate_documents_to_azure.sh
   ./scripts/migrate_documents_to_azure.sh
   ```

### Phase 4: Container Image Preparation

1. **Build and Push Docker Images:**
   ```bash
   # Login to Azure Container Registry
   az acr login --name miniragregistry
   
   # Build and push images using the Azure-specific compose file
   docker compose -f azure-docker-compose.yml build
   docker compose -f azure-docker-compose.yml push
   ```

### Phase 5: Deployment

1. **Create Container Apps Environment:**
   ```bash
   az containerapp env create \
     --name mini-rag-env \
     --resource-group mini-rag-rg \
     --location eastus
   ```

2. **Deploy Backend Container:**
   ```bash
   az containerapp create \
     --name mini-rag-backend \
     --resource-group mini-rag-rg \
     --environment mini-rag-env \
     --image ${ACR_SERVER}/mini-rag-backend:v1.0 \
     --target-port 8000 \
     --ingress external \
     --registry-server ${ACR_SERVER} \
     --registry-username ${ACR_USERNAME} \
     --registry-password ${ACR_PASSWORD} \
     --min-replicas 1 \
     --max-replicas 3 \
     --env-vars AZURE_OPENAI_ENDPOINT="https://mini-rag-openai.openai.azure.com" \
                AZURE_OPENAI_API_KEY=${OPENAI_KEY} \
                AZURE_OPENAI_DEPLOYMENT="gpt-4-turbo" \
                AZURE_SEARCH_ENDPOINT="https://mini-rag-search.search.windows.net" \
                AZURE_SEARCH_API_KEY=${SEARCH_KEY} \
                AZURE_SEARCH_INDEX="documents" \
                AZURE_STORAGE_CONNECTION_STRING=${STORAGE_CONNECTION}
   ```

3. **Deploy Frontend Container:**
   ```bash
   # Get backend FQDN
   BACKEND_FQDN=$(az containerapp show \
     --name mini-rag-backend \
     --resource-group mini-rag-rg \
     --query properties.configuration.ingress.fqdn -o tsv)
   
   # Deploy frontend
   az containerapp create \
     --name mini-rag-frontend \
     --resource-group mini-rag-rg \
     --environment mini-rag-env \
     --image ${ACR_SERVER}/mini-rag-frontend:v1.0 \
     --target-port 3000 \
     --ingress external \
     --registry-server ${ACR_SERVER} \
     --registry-username ${ACR_USERNAME} \
     --registry-password ${ACR_PASSWORD} \
     --min-replicas 1 \
     --max-replicas 3 \
     --env-vars REACT_APP_AZURE_ENABLED="true" \
                REACT_APP_API_URL="https://${BACKEND_FQDN}/api"
   ```

### Phase 6: Verification and Testing

1. **Retrieve Frontend URL:**
   ```bash
   FRONTEND_URL=$(az containerapp show \
     --name mini-rag-frontend \
     --resource-group mini-rag-rg \
     --query properties.configuration.ingress.fqdn -o tsv)
   
   echo "Application deployed at: https://${FRONTEND_URL}"
   ```

2. **Functional Testing Checklist:**
   - Document Upload Verification
   - Query Processing Validation
   - Source Attribution Testing
   - Response Latency Assessment
   - Error Handling Verification

## Resource Optimization

### Performance Tuning

1. **Add Azure Cache for Redis:**
   ```bash
   az redis create \
     --name mini-rag-cache \
     --resource-group mini-rag-rg \
     --location eastus \
     --sku Basic \
     --vm-size C0
   ```

2. **Configure Scaling Rules:**
   ```bash
   az containerapp update \
     --name mini-rag-backend \
     --resource-group mini-rag-rg \
     --scale-rule-name http-rule \
     --scale-rule-type http \
     --scale-rule-http-concurrency 20
   ```

3. **Implement Application Insights:**
   ```bash
   az monitor app-insights component create \
     --app mini-rag-insights \
     --location eastus \
     --resource-group mini-rag-rg
   ```

### Cost Management

1. **Reserved Instances:**
   For predictable workloads, consider Azure Reserved Instances for:
   - Container Apps compute reservations
   - Azure OpenAI capacity reservations

2. **Storage Tiering:**
   Implement lifecycle management for document storage:
   ```bash
   az storage account management-policy create \
     --account-name miniragstore \
     --resource-group mini-rag-rg \
     --policy @policy.json
   ```

3. **Usage Monitoring:**
   ```bash
   az monitor metrics alert create \
     --name "OpenAI Token Usage" \
     --resource-group mini-rag-rg \
     --scopes /subscriptions/{sub-id}/resourceGroups/mini-rag-rg/providers/Microsoft.CognitiveServices/accounts/mini-rag-openai \
     --condition "total token count > 10000" \
     --description "Alert when OpenAI token usage exceeds threshold"
     --evaluation-frequency 1h
   ```

## Rollback Procedures

### Immediate Rollback

If critical issues arise, execute the following rollback procedure:

1. **Revert to Local Environment:**
   ```bash
   # Start local deployment
   docker compose up -d
   
   # Update DNS or load balancer to point to local deployment
   # (environment-specific implementation)
   ```

2. **Traffic Redirection:**
   Update any external routing to direct traffic to the local environment.

### Phased Rollback

For less critical issues, consider component-specific rollback:

1. **Selective Service Reversion:**
   ```bash
   # Example: Revert LLM service while keeping other Azure services
   az containerapp update \
     --name mini-rag-backend \
     --resource-group mini-rag-rg \
     --env-vars USE_LOCAL_LLM="true" \
               AZURE_OPENAI_ENABLED="false"
   ```

## Maintenance Procedures

### Routine Maintenance

1. **Database Index Optimization:**
   ```bash
   # Optimize Azure Cognitive Search index
   az search service update \
     --name mini-rag-search \
     --resource-group mini-rag-rg \
     --replica-count 1
   ```

2. **Log Retention and Analysis:**
   ```bash
   # Configure diagnostic settings
   az monitor diagnostic-settings create \
     --name mini-rag-diag \
     --resource-group mini-rag-rg \
     --resource mini-rag-backend \
     --storage-account miniragstore \
     --logs '[{"category": "ContainerAppConsoleLogs", "enabled": true}]'
   ```

3. **Security Updates:**
   ```bash
   # Update container images with security patches
   docker compose -f azure-docker-compose.yml build --pull
   docker compose -f azure-docker-compose.yml push
   
   # Restart container apps to apply updates
   az containerapp update \
     --name mini-rag-backend \
     --resource-group mini-rag-rg \
     --image ${ACR_SERVER}/mini-rag-backend:v1.0
   ```

## Troubleshooting Guide

### Common Issues and Resolutions

| Issue | Diagnostic Steps | Resolution |
|-------|-----------------|------------|
| OpenAI Service Failures | Check API logs and quota limits | Increase capacity or implement request throttling |
| Search Indexing Errors | Verify index schema compatibility | Update schema definitions or reprocess documents |
| Container App Crashes | Review container logs and resource utilization | Adjust resource limits or optimize application code |
| Document Processing Failures | Examine process logs and file metadata | Add format validation or enhance error handling |

### Diagnostic Commands

```bash
# Get container app logs
az containerapp logs show \
  --name mini-rag-backend \
  --resource-group mini-rag-rg \
  --follow

# Test Azure OpenAI connectivity
curl -X POST \
  "https://mini-rag-openai.openai.azure.com/openai/deployments/gpt-4-turbo/chat/completions?api-version=2023-05-15" \
  -H "Content-Type: application/json" \
  -H "api-key: ${OPENAI_KEY}" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Verify Cognitive Search index status
az search index show \
  --name documents \
  --service-name mini-rag-search \
  --resource-group mini-rag-rg
```

## Conclusion

This migration guide provides a comprehensive framework for transitioning the Mini-RAG system from local development to Azure cloud services. By following these structured implementation steps, you can achieve a robust, scalable deployment with optimized resource utilization and maintainable architecture.

## Appendices

### Resource Configuration Templates

See the `azure/templates` directory for complete ARM and Bicep templates.

### Monitoring Dashboard Setup

Refer to `azure/monitoring` for dashboard configuration files and query templates.

### Security Considerations

Review `azure/security` for detailed security configuration guidelines and compliance documentation.