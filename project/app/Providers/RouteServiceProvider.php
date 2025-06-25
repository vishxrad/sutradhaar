<?php

namespace App\Providers;

// use Illuminate\Support\ServiceProvider;


// namespace App\Providers;

use Illuminate\Support\Facades\Route; // ✅ Add this line
use Illuminate\Foundation\Support\Providers\RouteServiceProvider as ServiceProvider;
use Illuminate\Support\Facades\Schema;

class RouteServiceProvider extends ServiceProvider
{
    // ...


    /**
     * Register services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap services.
     */
    
    public function boot()
{
    Route::middleware('api')
        ->prefix('api')
        ->group(base_path('routes/api.php'));

    Route::middleware('web')
        ->group(base_path('routes/web.php'));
}

}
