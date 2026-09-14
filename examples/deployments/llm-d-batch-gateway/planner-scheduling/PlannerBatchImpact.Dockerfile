# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/ai-dynamo/dynamo-planner-nightly:latest@sha256:82f1952370e6c3225f1955740d767d0ea39fdd85145d160a259e5d951fb19399

ARG DYNAMO_COMMIT_SHA
ARG PLANNER_SOURCE_SHA256

RUN uv pip install --python /opt/dynamo/venv/bin/python "redis>=6.2.0,<9.0.0"

COPY --chown=1000:0 components/src/dynamo/planner /workspace/components/src/dynamo/planner

ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

LABEL org.opencontainers.image.base.name="nvcr.io/nvidia/ai-dynamo/dynamo-planner-nightly:latest" \
      org.opencontainers.image.base.digest="sha256:82f1952370e6c3225f1955740d767d0ea39fdd85145d160a259e5d951fb19399" \
      org.opencontainers.image.revision="${DYNAMO_COMMIT_SHA}" \
      io.dynamo.poc.variant="planner-gym-batch-impact" \
      io.dynamo.poc.planner-source-sha256="${PLANNER_SOURCE_SHA256}"
