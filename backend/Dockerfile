# Stage 1: Pull the models
FROM ollama/ollama:latest as builder

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434

# Install necessary dependencies
RUN apt-get update && apt-get install -y curl

# Start Ollama server in the background
RUN mkdir -p /root/.ollama
RUN nohup ollama serve > /var/log/ollama.log 2>&1 & \
    # Wait for Ollama server to start
    sleep 10 && \
    # Pull both models
    ollama pull deepseek-r1:8b && \
    ollama pull llama3.2:3b && \
    # Wait for models to be fully downloaded
    sleep 60

# Stage 2: Create the final image
FROM ollama/ollama:latest

# Set environment variables for Ollama
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_ORIGINS=*

# Copy the downloaded models from the builder stage
COPY --from=builder /root/.ollama /root/.ollama

# Expose the default Ollama port
EXPOSE 11434

# Persist models locally
VOLUME ["/local/repository:/root/.ollama"]

# Add a volume for /data
VOLUME /data

# Set the entrypoint
ENTRYPOINT ["ollama"]
CMD ["serve"]