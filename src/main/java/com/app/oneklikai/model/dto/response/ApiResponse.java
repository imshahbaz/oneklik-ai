package com.app.oneklikai.model.dto.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class ApiResponse<T> {

    boolean success;
    String message;
    T data;
    String error;

    public static <T> ApiResponse<T> ok(T data, String message) {
        return ApiResponse.<T>builder()
                .success(true)
                .message(message)
                .data(data)
                .build();
    }

    public static <T> ApiResponse<T> fail(String errorMessage) {
        return ApiResponse.<T>builder()
                .success(false)
                .error(errorMessage)
                .build();
    }

}