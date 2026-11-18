import os
from celery import Celery
from core.config import settings

# Initialize Celery with Redis broker and result backend
celery_app = Celery(
    "docusense_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=False,
    broker_connection_max_retries=2,
    broker_transport_options={
        "max_retries": 2,
        "interval_start": 0,
        "interval_step": 0.1,
        "interval_max": 0.2,
    },
    # Worker concurrency and rate limit configurations
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
