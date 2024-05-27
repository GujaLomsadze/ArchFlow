# Use the official Grafana base image
FROM grafana/grafana:latest

# Install the Redis datasource plugin
RUN grafana-cli --pluginsDir "/var/lib/grafana/plugins" plugins install redis-datasource
