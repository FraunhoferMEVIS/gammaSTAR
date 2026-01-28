FROM python:3.12 AS build-base

# Install dependencies
RUN apt update

# Build wheel 
FROM build-base AS build

FROM python:3.12 AS server-base
RUN pip install fastapi[standard]

FROM server-base AS server
ADD gammastar_recon/mrpy_tools/reconstruction /src/reconstruction
RUN pip install -e /src/reconstruction/mrpy_recon_tools
ADD ./gammastar_recon /src/gammastar_recon

# Add sequenceServer
RUN pip install ismrmrd 
RUN pip install h5py
RUN pip install pydicom
COPY gammastar_ilumr/sequenceServer.py /src/
RUN mkdir /data_export

# Expose the ports for both endpoints
EXPOSE 8900

# Start both FastAPI applications
CMD ["fastapi", "run", "/src/sequenceServer.py", "--port", "8900", "--host", "0.0.0.0"]