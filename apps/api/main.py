from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from apps.api.config import settings