package com.Java.fraud_detection_api.Controller;

import com.Java.fraud_detection_api.Service.FraudDetectionService;
import com.Java.fraud_detection_api.Dto.TransactionRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class FraudDetectionController {

    @Autowired
    private FraudDetectionService fraudDetectionService;

    // ✅ GET endpoint for quick testing
    // Supports both "hourOfDay" and "hour" as query params
    @GetMapping("/check")
    public String checkTransactionGet(
            @RequestParam Double amount,
            @RequestParam(name = "hourOfDay", required = false) Integer hourOfDay,
            @RequestParam(name = "hour", required = false) Integer hour) {

        // Use whichever is provided (prefer hourOfDay if both exist)
        Integer finalHour = (hourOfDay != null) ? hourOfDay : hour;

        if (finalHour == null) {
            return "❌ Missing required parameter: hourOfDay or hour";
        }

        return fraudDetectionService.checkTransaction(amount, finalHour);
    }

    // ✅ POST endpoint for production use
    @PostMapping("/check")
    public String checkTransactionPost(@RequestBody TransactionRequest request) {
        return fraudDetectionService.checkTransaction(request);
    }
}
