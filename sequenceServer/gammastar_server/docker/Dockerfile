FROM python:3.12 AS build-base

# Install dependencies
RUN apt update
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install maturin

# Build wheel 
FROM build-base AS build
ADD gammastar_parser/gstar-parser /src/gstar-parser
ADD gammastar_parser/pygstar /src/pygstar
COPY gammastar_parser/custom_pygstar_cargo.toml /src/pygstar/Cargo.toml
WORKDIR /src/pygstar
RUN maturin build --interpreter python

FROM python:3.12 AS server-base
RUN pip install fastapi[standard]
RUN pip install MRzeroCore==0.3.6
COPY gammastar_server/generatePhantoms.py /src/
RUN curl -L https://raw.githubusercontent.com/MRsources/MRzero-Core/v0.3.5/python/MRzeroCore/phantom/brainweb/brainweb_data.json -o /tmp/brainweb_data.json && \
    awk 'NR==23{$0="    4"} {print}' /tmp/brainweb_data.json > /usr/local/lib/python3.12/site-packages/MRzeroCore/phantom/brainweb/brainweb_data.json
RUN python /src/generatePhantoms.py

FROM server-base AS server
COPY --from=build /src/pygstar/target/wheels/pygstar-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl /
RUN pip install /pygstar-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl

RUN mv /usr/local/lib/python3.12/site-packages/MRzeroCore/sequence.py /usr/local/lib/python3.12/site-packages/MRzeroCore/sequence.bak.py
RUN curl -L https://raw.githubusercontent.com/MRsources/MRzero-Core/v0.3.5/python/MRzeroCore/sequence.py -o /tmp/sequence.py && \
    awk 'NR==8{print "import pygstar"} {print}' /tmp/sequence.py > /tmp/sequence_mod1.py && \
    awk 'NR==444{print "        parser = pygstar.load_gstar(file_name)"} NR!=444{print}' /tmp/sequence_mod1.py > /usr/local/lib/python3.12/site-packages/MRzeroCore/sequence.py
ADD gammastar_recon/mrpy_tools/reconstruction /src/reconstruction
RUN pip install -e /src/reconstruction/mrpy_recon_tools
ADD ./gammastar_recon /src/gammastar_recon

# Add sequenceServer
RUN pip install ismrmrd 
RUN pip install h5py
RUN pip install pydicom
COPY gammastar_server/sequenceServer.py /src/
RUN mkdir /data_export

# Expose the ports for both endpoints
EXPOSE 8800

# Start both FastAPI applications
CMD ["fastapi", "run", "/src/sequenceServer.py", "--port", "8800", "--host", "0.0.0.0"]
