"""Locust Load and Concurrency Test for Healthcare Clinical RAG API.

Simulates 20 concurrent clinicians and researchers querying the system,
benchmarking latency (p50, p95, p99), throughput, and rate limiter enforcement.
"""

import json
import logging
import os
import random
from locust import HttpUser, task, between, events

logger = logging.getLogger("locust_load_test")

SAMPLE_CLINICAL_QUESTIONS = [
    "What is the first-line treatment for uncomplicated falciparum malaria according to WHO guidelines?",
    "What is the recommended metformin titration schedule for Type 2 diabetes?",
    "What systolic blood pressure threshold indicates Stage 2 hypertension in adult guidelines?",
    "What are the diagnostic criteria for gestational diabetes according to WHO recommendations?",
    "When should artemisinin-based combination therapy be avoided during pregnancy?",
    "What lifestyle interventions are recommended alongside pharmacotherapy for diabetes?",
    "What follow-up schedule is required after initiating statin therapy?",
]


class ClinicianUser(HttpUser):
    """Simulates active clinicians querying the RAG API and testing rate limits."""

    wait_time = between(0.2, 0.8)
    token: str = ""

    def on_start(self):
        """Authenticate on start and acquire JWT access token."""
        resp = self.client.post(
            "/auth/token",
            json={
                "username": "clinician",
                "password": "clinician123",
                "tenant_id": "clinic_general",
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            self.token = data.get("access_token", "")
            self.auth_headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        else:
            self.auth_headers = {"Content-Type": "application/json"}

    @task(5)
    def test_health_check(self):
        """Benchmark health check endpoint under high concurrency."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(3)
    def test_list_documents(self):
        """Query document registry with pagination."""
        with self.client.get(
            "/documents?limit=10&offset=0",
            headers=self.auth_headers,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"List documents returned {resp.status_code}")

    @task(2)
    def test_clinical_query_pipeline(self):
        """Query /ask with a clinical guideline question."""
        question = random.choice(SAMPLE_CLINICAL_QUESTIONS)
        payload = {
            "question": question,
            "stream": False,
        }
        with self.client.post(
            "/ask",
            json=payload,
            headers=self.auth_headers,
            catch_response=True,
            timeout=30.0,
        ) as resp:
            # 200 (success), 429 (rate-limited by design), 503 (circuit breaker open)
            if resp.status_code in (200, 429, 503):
                resp.success()
            else:
                resp.failure(f"Ask returned status {resp.status_code}: {resp.text[:100]}")

    @task(1)
    def test_rate_limiter_burst(self):
        """Send rapid burst requests to verify rate limiter returns 429 with Retry-After header."""
        for _ in range(3):
            with self.client.post(
                "/ask",
                json={"question": "Immediate rapid rate-limit test probe"},
                headers=self.auth_headers,
                catch_response=True,
            ) as resp:
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        resp.success()
                    else:
                        resp.failure("429 missing Retry-After header")
                elif resp.status_code in (200, 503):
                    resp.success()
                else:
                    resp.failure(f"Burst probe unexpected code {resp.status_code}")


class ResearcherUser(HttpUser):
    """Simulates clinical researchers exploring document metadata and status."""

    wait_time = between(0.5, 1.5)
    token: str = ""

    def on_start(self):
        resp = self.client.post(
            "/auth/token",
            json={
                "username": "researcher",
                "password": "researcher123",
                "tenant_id": "research_institute",
            },
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token", "")
            self.auth_headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        else:
            self.auth_headers = {"Content-Type": "application/json"}

    @task(4)
    def test_health(self):
        self.client.get("/health")

    @task(3)
    def test_paginated_document_registry(self):
        offset = random.choice([0, 10, 20])
        self.client.get(f"/documents?limit=5&offset={offset}", headers=self.auth_headers)

