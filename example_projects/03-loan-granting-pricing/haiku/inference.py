"""Inference handler for Flex Loan granting."""

from decider import RequestHandler


class FlexLoanInferenceHandler(RequestHandler):
    """Handle inference requests for Flex Loan granting."""

    def invoke(self, request):
        """Process a granting request and return decision."""
        try:
            result = self.pipeline(request)
            return {
                "decision_id": result.get("decision_id"),
                "outcome": result.get("outcome"),
                "offers": result.get("validated_offers", []),
                "reason_codes": result.get("reason_codes", []),
                "primary_reason_code": result.get("primary_reason_code"),
            }
        except Exception as e:
            return {
                "error": str(e),
                "outcome": "error",
            }
