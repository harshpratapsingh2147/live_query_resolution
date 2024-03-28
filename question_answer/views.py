from rest_framework.generics import GenericAPIView
from .utility import question_answer
from .validate_serializer import LiveQueryValidateSerializer
from rest_framework.response import Response
# Create your views here.


class LiveQuestionAnswer(GenericAPIView):
    validate_serializer_class = LiveQueryValidateSerializer

    def get(self, request):
        class_id = request.GET.get('class_id')
        query = request.GET.get('query')
        ask_expert = int(request.GET.get('ask_expert'))
        filter_serializer = self.validate_serializer_class(data=request.GET)

        if not filter_serializer.is_valid():
            return Response(filter_serializer.errors)

        if ask_expert:
            result = """We have notified out expert. 
            Your query will be answered in 24 hours.
            Thanks for asking!"""
        else:
            result = question_answer(class_id=class_id, query=query)

        res = {
            query: result
        }
        return Response(res)


