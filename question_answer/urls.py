
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import LiveQuestionAnswer

router = DefaultRouter()

urlpatterns = [
    path("", include(router.urls)),
    path(
        "query", LiveQuestionAnswer.as_view(), name="live_question_answer"
    )
]