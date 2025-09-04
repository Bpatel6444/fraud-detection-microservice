package com.Java.fraud_detection_api.Controller;

import com.Java.fraud_detection_api.Service.FraudDetectionService;
import com.Java.fraud_detection_api.Dto.TransactionRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.Map;

@RestController
@RequestMapping("/api/fraud")
public class FraudDetectionController {

    @Autowired
    private FraudDetectionService fraudDetectionService;

    // GET endpoint for quick testing
    @GetMapping("/check")
    public String checkTransactionGet(
            @RequestParam Double amount,
            @RequestParam String timestamp,
            @RequestParam(required = false) String transactionId,
            @RequestParam(required = false) String merchantId,
            @RequestParam(required = false) String customerId,
            @RequestParam(required = false) Map<String, Double> pcaFeatures) {
        
        Instant instant = Instant.parse(timestamp);
        return fraudDetectionService.checkTransaction(amount, instant, transactionId, merchantId, customerId, pcaFeatures);
    }

    // POST endpoint for production use
    @PostMapping("/check")
    public String checkTransactionPost(@RequestBody TransactionRequest request) {
        return fraudDetectionService.checkTransaction(request);
    }
    
    // Health check endpoint
    @GetMapping("/health")
    public String healthCheck() {
        return "{\"status\": \"OK\", \"service\": \"Fraud Detection API\", \"version\": \"2.0\"}";
    }
    
    // Model info endpoint
    @GetMapping("/model-info")
    public String modelInfo() {
        return "{\"model\": \"XGBoost/LightGBM Ensemble\", \"features\": \"PCA + Time + Amount\", \"version\": \"enhanced\"}";
    }
}