package com.Java.fraud_detection_api.Service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
public class MlClientService {

    @Autowired
    private RestTemplate restTemplate;

    // Base URL of the Python ML service
    private static final String ML_API_URL = "http://localhost:5000/predict";

    public String checkTransactionWithML(Double amount, Integer hourOfDay) {
        try {
            // Build JSON request body
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("amount", amount);
            requestBody.put("hourOfDay", hourOfDay);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

            // Send POST request to Python ML service
            ResponseEntity<String> response = restTemplate.postForEntity(ML_API_URL, request, String.class);

            return response.getBody();  // Return Python service response directly

        } catch (Exception e) {
            return "{\"error\": \"ML service unavailable: " + e.getMessage() + "\"}";
        }
    }
}
