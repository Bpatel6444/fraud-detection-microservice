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

    // GET endpoint for quick testing
    @GetMapping("/check")
    public String checkTransactionGet(
            @RequestParam Double amount,
            @RequestParam Integer hourOfDay) {
        return fraudDetectionService.checkTransaction(amount, hourOfDay);
    }

    // POST endpoint for production use
    @PostMapping("/check")
    public String checkTransactionPost(
            @RequestBody TransactionRequest request) {
        return fraudDetectionService.checkTransaction(request);
    }
}