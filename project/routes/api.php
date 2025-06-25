<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ScriptsController;

Route::post('/save-script', [ScriptsController::class, 'saveOrUpdate']);

// Optional: keep this test route too
Route::get('/check-api', function () {
    return response()->json(['status' => 'API is working!']);
});
