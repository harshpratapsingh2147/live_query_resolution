from rest_framework import serializers
from .utility import valid_integer


class LiveQueryValidateSerializer(serializers.Serializer):
    class_id = serializers.CharField(required=True)
    query = serializers.CharField(required=True)
    ask_expert = serializers.CharField(required=True)

    def validate_class_id(self, value):
        if not valid_integer(value):
            raise serializers.ValidationError(
                "class_id can only be integer"
            )
        return value

    def validate_ask_expert(self, value):
        if value not in ['0', '1']:
            raise serializers.ValidationError(
                "ask_expert can only be 0 or 1"
            )


