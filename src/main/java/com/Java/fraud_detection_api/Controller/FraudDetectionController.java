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

    // GET endpoint for quick rule-based test
    @GetMapping("/check")
    public String checkTransactionGet(
            @RequestParam Double amount,
            @RequestParam(name = "hourOfDay", required = false) Integer hourOfDay,
            @RequestParam(name = "hour", required = false) Integer hour) {

        Integer finalHour = (hourOfDay != null) ? hourOfDay : hour;

        if (finalHour == null) {
            return "❌ Missing required parameter: hourOfDay or hour";
        }

        return fraudDetectionService.checkTransaction(amount, finalHour);
    }

    // POST endpoint for ML-based check
    @PostMapping("/check")
    public String checkTransactionPost(@RequestBody TransactionRequest request) {
        return fraudDetectionService.checkTransaction(request);  // ML service called here
    }
}
